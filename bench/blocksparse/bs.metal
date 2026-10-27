// Flash attention (bf16 in/out, fp32 softmax/accum), head-major [H][S][D] in and out,
// matching h3_gpu_sdpa_bf16_head_major_output with headMajorSDPAInputs (DiT path).
// Each simdgroup owns RQ query rows; streams K/V in BK-row blocks; QK^T and PV via
// MPP matmul2d (execution_simdgroup) like ~/ai/h3/bench/int8peak/k.metal.
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
// Block-sparse variant of the flash kernel (bench/flashattn, h3.c-reuse-flashattn-1025 29322b3):
// FastVideo VSA fine branch = attention restricted to selected 64-token key tiles
// (video_sparse_attn_h3.py _build_block_mask #L572-619). sel[head][query_tile][NSEL] lists key
// tile indices; sequence must already be in tile order (64 rows per tile).
struct bs_args { uint S, H; float scale_log2; uint QT, NSEL; };
constant constexpr uint D = 128;

template<uint RQ, uint BK, uint SG, bool QREG>
kernel void bs_bf16(device const bfloat *Q [[buffer(0)]], device const bfloat *K [[buffer(1)]],
                    device const bfloat *V [[buffer(2)]], device bfloat *O [[buffer(3)]],
                    constant bs_args &a [[buffer(4)]], device const uint *sel [[buffer(5)]],
                    uint2 g [[threadgroup_position_in_grid]],
                    ushort sg [[simdgroup_index_in_threadgroup]],
                    ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup bfloat Pbuf[SG][RQ*BK];
    threadgroup float rmax[SG][RQ], rsum[SG][RQ], mrow[SG][RQ], lrow[SG][RQ], alpha[SG][RQ];
    const uint S = a.S;
    const uint head = g.y;
    const uint row0 = (g.x * SG + sg) * RQ;
    if (row0 >= S) return;  // whole simdgroup uniform
    size_t hoff = (size_t)head * S * D;
    auto q = tensor<device bfloat, dextents<int32_t,2>, tensor_inline>((device bfloat*)Q + hoff, dextents<int32_t,2>(D, S));
    auto k = tensor<device bfloat, dextents<int32_t,2>, tensor_inline>((device bfloat*)K + hoff, dextents<int32_t,2>(D, S));
    auto v = tensor<device bfloat, dextents<int32_t,2>, tensor_inline>((device bfloat*)V + hoff, dextents<int32_t,2>(D, S));
    auto p = tensor<threadgroup bfloat, dextents<int32_t,2>, tensor_inline>(&Pbuf[sg][0], dextents<int32_t,2>(BK, RQ));

    constexpr auto dS = matmul2d_descriptor(RQ, BK, D, false, true, true, matmul2d_descriptor::mode::multiply);
    constexpr auto dO = matmul2d_descriptor(RQ, D, BK, false, false, true, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<dS, execution_simdgroup> mmS;
    matmul2d<dO, execution_simdgroup> mmO;

    auto qa = q.template slice<D, RQ>(0, (int)row0);
    auto kb0 = k.template slice<D, BK>(0, 0);
    auto pa = p.template slice<BK, RQ>(0, 0);
    auto vb0 = v.template slice<D, BK>(0, 0);
    auto sct = mmS.template get_destination_cooperative_tensor<decltype(qa), decltype(kb0), float>();
    auto red = mmS.template get_row_reduction_destination_cooperative_tensor<decltype(qa), decltype(kb0), float>();
    auto qct = mmS.template get_left_input_cooperative_tensor<bfloat, bfloat, float>();
    if (QREG) qct.load(qa);
    auto oct = mmO.template get_destination_cooperative_tensor<decltype(pa), decltype(vb0), float>();
    #pragma clang loop unroll(full)
    for (ushort e = 0; e < oct.get_capacity(); e++) if (oct.is_valid_element(e)) oct[e] = 0;
    for (uint r = lane; r < RQ; r += 32) { mrow[sg][r] = -INFINITY; lrow[sg][r] = 0; }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    const float sl2 = a.scale_log2;

    device const uint *list = sel + ((size_t)head * a.QT + g.x) * a.NSEL;
    const uint count = list[0];  // row layout: [count, tile0, tile1, ...], stride NSEL
    for (uint it = 0; it < count * (64 / BK); it++) {
        uint kb = list[1 + it / (64 / BK)] * 64 + (it % (64 / BK)) * BK;
        auto kt = k.template slice<D, BK>(0, (int)kb);
        if (QREG) mmS.run(qct, kt, sct); else mmS.run(qa, kt, sct);
        const bool tail = kb + BK > S;
        #pragma clang loop unroll(full)
        for (ushort e = 0; e < sct.get_capacity(); e++) {
            if (!sct.is_valid_element(e)) continue;
            float s = sct[e] * sl2;
            if (tail) { auto i = sct.get_multidimensional_index(e); if (kb + (uint)i[0] >= S) s = -INFINITY; }
            sct[e] = s;
        }
        reduce_rows(sct, red, reduction_operation::max, -INFINITY);
        #pragma clang loop unroll(full)
        for (ushort e = 0; e < red.get_capacity(); e++) {
            if (!red.is_valid_element(e)) continue;
            auto i = red.get_multidimensional_index(e);
            rmax[sg][i[0]] = red[e];
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (uint r = lane; r < RQ; r += 32) {
            float mo = mrow[sg][r], mn = max(mo, rmax[sg][r]);
            alpha[sg][r] = exp2(mo - mn);   // mo=-inf first block -> 0
            mrow[sg][r] = mn;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        #pragma clang loop unroll(full)
        for (ushort e = 0; e < sct.get_capacity(); e++) {
            if (!sct.is_valid_element(e)) continue;
            auto i = sct.get_multidimensional_index(e);
            float pv = exp2(sct[e] - mrow[sg][i[1]]);
            sct[e] = pv;
            Pbuf[sg][(uint)i[1] * BK + (uint)i[0]] = bfloat(pv);
        }
        reduce_rows(sct, red, reduction_operation::sum, 0.0f);
        #pragma clang loop unroll(full)
        for (ushort e = 0; e < red.get_capacity(); e++) {
            if (!red.is_valid_element(e)) continue;
            auto i = red.get_multidimensional_index(e);
            rsum[sg][i[0]] = red[e];
        }
        #pragma clang loop unroll(full)
        for (ushort e = 0; e < oct.get_capacity(); e++) {
            if (!oct.is_valid_element(e)) continue;
            auto i = oct.get_multidimensional_index(e);
            oct[e] *= alpha[sg][i[1]];
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (uint r = lane; r < RQ; r += 32) lrow[sg][r] = lrow[sg][r] * alpha[sg][r] + rsum[sg][r];
        auto vt = v.template slice<D, BK>(0, (int)kb);
        mmO.run(pa, vt, oct);
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    #pragma clang loop unroll(full)
    for (ushort e = 0; e < oct.get_capacity(); e++) {
        if (!oct.is_valid_element(e)) continue;
        auto i = oct.get_multidimensional_index(e);
        uint r = row0 + (uint)i[1];
        if (r < S) O[hoff + (size_t)r * D + (uint)i[0]] = bfloat(oct[e] / lrow[sg][i[1]]);
    }
}
template [[host_name("bs_r16_k64_s4")]] kernel void bs_bf16<16,64,4,false>(
  device const bfloat*,device const bfloat*,device const bfloat*,device bfloat*,constant bs_args&,device const uint*,uint2,ushort,ushort);
