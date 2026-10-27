#include "h3_gpu.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static unsigned short to_bf16(float value) {
    union { float f; unsigned u; } bits = {value};
    bits.u += 0x7fffu + ((bits.u >> 16) & 1u);
    return (unsigned short)(bits.u >> 16);
}

static float from_bf16(unsigned short value) {
    union { unsigned u; float f; } bits = {(unsigned)value << 16};
    return bits.f;
}

static float urand(void) {
    return ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
}

static void cpu_coarse(float *out, const float *q, const float *k,
                       const float *v, const float *gate, const unsigned *map,
                       unsigned rows, unsigned heads, unsigned dim,
                       unsigned tiles, float scale) {
    for (unsigned row = 0; row < rows; row++) {
        for (unsigned head = 0; head < heads; head++) {
            float scores[16] = {0};
            float values[16] = {0};
            float max_score = -INFINITY;
            for (unsigned tile = 0; tile < tiles; tile++) {
                float score = 0.0f;
                for (unsigned d = 0; d < dim; d++) {
                    float qm = 0.0f, km = 0.0f;
                    unsigned qc = 0, kc = 0;
                    for (unsigned r = 0; r < rows; r++) {
                        if (map[r] == map[row]) {
                            qm += q[(r * heads + head) * dim + d];
                            qc++;
                        }
                        if (map[r] == tile) {
                            km += k[(r * heads + head) * dim + d];
                            kc++;
                        }
                    }
                    score += (qm / (float)qc) * (km / (float)kc);
                }
                scores[tile] = score * scale;
                if (scores[tile] > max_score) max_score = scores[tile];
            }
            float denominator = 0.0f;
            for (unsigned tile = 0; tile < tiles; tile++)
                denominator += expf(scores[tile] - max_score);
            for (unsigned tile = 0; tile < tiles; tile++) {
                unsigned count = 0;
                float mean = 0.0f;
                for (unsigned r = 0; r < rows; r++)
                    if (map[r] == tile) {
                        mean += v[(r * heads + head) * dim];
                        count++;
                    }
                values[tile] = mean / (float)count;
            }
            for (unsigned d = 0; d < dim; d++) {
                float coarse = 0.0f;
                for (unsigned tile = 0; tile < tiles; tile++) {
                    unsigned count = 0;
                    float mean = 0.0f;
                    for (unsigned r = 0; r < rows; r++)
                        if (map[r] == tile) {
                            mean += v[(r * heads + head) * dim + d];
                            count++;
                        }
                    if (count)
                        coarse += expf(scores[tile] - max_score) /
                            denominator * mean / (float)count;
                }
                size_t index = (size_t)(row * heads + head) * dim + d;
                out[index] += gate[index] * coarse;
            }
        }
    }
}

/* Same math as cpu_coarse, pooled first (FastVideo
 * video_sparse_attn_h3.py:_pool_tiles), so it is cheap at real sizes. */
static void cpu_coarse_pooled(float *out, const float *q, const float *k,
                              const float *v, const float *gate,
                              const unsigned *map, unsigned rows,
                              unsigned heads, unsigned dim, unsigned tiles,
                              float scale) {
    size_t te = (size_t)tiles * heads * dim;
    double *qp = calloc(te, sizeof(*qp)), *kp = calloc(te, sizeof(*kp));
    double *vp = calloc(te, sizeof(*vp)), *oc = calloc(te, sizeof(*oc));
    unsigned *count = calloc(tiles, sizeof(*count));
    double *score = calloc(tiles, sizeof(*score));
    if (!qp || !kp || !vp || !oc || !count || !score) exit(2);
    for (unsigned r = 0; r < rows; r++) {
        count[map[r]]++;
        for (unsigned h = 0; h < heads; h++)
            for (unsigned d = 0; d < dim; d++) {
                size_t src = ((size_t)r * heads + h) * dim + d;
                size_t dst = ((size_t)map[r] * heads + h) * dim + d;
                qp[dst] += q[src]; kp[dst] += k[src]; vp[dst] += v[src];
            }
    }
    for (unsigned t = 0; t < tiles; t++)
        for (size_t i = 0; i < (size_t)heads * dim && count[t]; i++) {
            size_t j = (size_t)t * heads * dim + i;
            qp[j] /= count[t]; kp[j] /= count[t]; vp[j] /= count[t];
        }
    for (unsigned t = 0; t < tiles; t++) {
        if (!count[t]) continue;
        for (unsigned h = 0; h < heads; h++) {
            const double *qv = qp + ((size_t)t * heads + h) * dim;
            double best = -INFINITY, sum = 0.0;
            for (unsigned u = 0; u < tiles; u++) {
                const double *kv = kp + ((size_t)u * heads + h) * dim;
                double dot = 0.0;
                for (unsigned d = 0; d < dim; d++) dot += qv[d] * kv[d];
                score[u] = count[u] ? dot * scale : -INFINITY;
                if (score[u] > best) best = score[u];
            }
            for (unsigned u = 0; u < tiles; u++) {
                score[u] = count[u] ? exp(score[u] - best) : 0.0;
                sum += score[u];
            }
            double *o = oc + ((size_t)t * heads + h) * dim;
            for (unsigned u = 0; u < tiles; u++)
                for (unsigned d = 0; d < dim; d++)
                    o[d] += score[u] / sum *
                        vp[((size_t)u * heads + h) * dim + d];
        }
    }
    for (unsigned r = 0; r < rows; r++)
        for (unsigned h = 0; h < heads; h++)
            for (unsigned d = 0; d < dim; d++) {
                size_t i = ((size_t)r * heads + h) * dim + d;
                out[i] += gate[i] *
                    (float)oc[((size_t)map[r] * heads + h) * dim + d];
            }
    free(qp); free(kp); free(vp); free(oc); free(count); free(score);
}

static void to_head_major(unsigned short *dst, const unsigned short *src,
                          unsigned rows, unsigned heads, unsigned dim) {
    for (unsigned r = 0; r < rows; r++)
        for (unsigned h = 0; h < heads; h++)
            memcpy(dst + ((size_t)h * rows + r) * dim,
                   src + ((size_t)r * heads + h) * dim,
                   dim * sizeof(*dst));
}

static void from_head_major(unsigned short *dst, const unsigned short *src,
                            unsigned rows, unsigned heads, unsigned dim) {
    for (unsigned r = 0; r < rows; r++)
        for (unsigned h = 0; h < heads; h++)
            memcpy(dst + ((size_t)r * heads + h) * dim,
                   src + ((size_t)h * rows + r) * dim,
                   dim * sizeof(*dst));
}

/* Runs the Metal coarse branch on random data and returns the max error
 * against the CPU reference (slow reference when pooled_reference is 0). */
static int run_case(h3_gpu *gpu, unsigned rows, unsigned heads, unsigned dim,
                    const unsigned *map, unsigned tiles, int qkv_head_major,
                    int output_head_major, int pooled_reference,
                    int repeats, float *max_error_out,
                    double *gpu_ms_out) {
    const size_t elements = (size_t)rows * heads * dim;
    float *q = malloc(elements * sizeof(*q));
    float *k = malloc(elements * sizeof(*k));
    float *v = malloc(elements * sizeof(*v));
    float *gate = malloc(elements * sizeof(*gate));
    float *expected = calloc(elements, sizeof(*expected));
    unsigned short *qb = malloc(elements * sizeof(*qb));
    unsigned short *kb = malloc(elements * sizeof(*kb));
    unsigned short *vb = malloc(elements * sizeof(*vb));
    unsigned short *gb = malloc(elements * sizeof(*gb));
    unsigned short *ob = malloc(elements * sizeof(*ob));
    unsigned short *tmp = malloc(elements * sizeof(*tmp));
    unsigned short *actual = malloc(elements * sizeof(*actual));
    unsigned *offsets = calloc((size_t)tiles + 1, sizeof(*offsets));
    unsigned *cursor = calloc((size_t)tiles + 1, sizeof(*cursor));
    unsigned *grouped = calloc(rows, sizeof(*grouped));
    if (!q || !k || !v || !gate || !expected || !qb || !kb || !vb || !gb ||
        !ob || !tmp || !actual || !offsets || !cursor || !grouped)
        return 0;
    srand(7);
    for (size_t i = 0; i < elements; i++) {
        q[i] = urand(); k[i] = urand(); v[i] = urand(); gate[i] = urand();
        qb[i] = to_bf16(q[i]); kb[i] = to_bf16(k[i]);
        vb[i] = to_bf16(v[i]); gb[i] = to_bf16(gate[i]);
        q[i] = from_bf16(qb[i]); k[i] = from_bf16(kb[i]);
        v[i] = from_bf16(vb[i]); gate[i] = from_bf16(gb[i]);
    }
    for (unsigned r = 0; r < rows; r++) offsets[map[r] + 1]++;
    for (unsigned t = 0; t < tiles; t++) offsets[t + 1] += offsets[t];
    memcpy(cursor, offsets, ((size_t)tiles + 1) * sizeof(*cursor));
    for (unsigned r = 0; r < rows; r++) grouped[cursor[map[r]]++] = r;
    float scale = 1.0f / sqrtf((float)dim);
    if (pooled_reference)
        cpu_coarse_pooled(expected, q, k, v, gate, map, rows, heads, dim,
                          tiles, scale);
    else
        cpu_coarse(expected, q, k, v, gate, map, rows, heads, dim, tiles,
                   scale);
    /* The dense SDPA output stand-in is q itself. */
    memcpy(ob, qb, elements * sizeof(*ob));
    h3_gpu_tensor *qt, *kt, *vt, *ot;
    if (qkv_head_major) {
        to_head_major(tmp, qb, rows, heads, dim);
        qt = h3_gpu_tensor_from_bf16(gpu, tmp, elements);
        to_head_major(tmp, kb, rows, heads, dim);
        kt = h3_gpu_tensor_from_bf16(gpu, tmp, elements);
        to_head_major(tmp, vb, rows, heads, dim);
        vt = h3_gpu_tensor_from_bf16(gpu, tmp, elements);
    } else {
        qt = h3_gpu_tensor_from_bf16(gpu, qb, elements);
        kt = h3_gpu_tensor_from_bf16(gpu, kb, elements);
        vt = h3_gpu_tensor_from_bf16(gpu, vb, elements);
    }
    if (output_head_major) to_head_major(tmp, ob, rows, heads, dim);
    else memcpy(tmp, ob, elements * sizeof(*tmp));
    ot = h3_gpu_tensor_from_bf16(gpu, tmp, elements);
    h3_gpu_tensor *gt = h3_gpu_tensor_from_bf16(gpu, gb, elements);
    h3_gpu_tensor *mt = h3_gpu_tensor_from_u32(gpu, map, rows);
    h3_gpu_tensor *rt = h3_gpu_tensor_from_u32(gpu, grouped, rows);
    h3_gpu_tensor *ft = h3_gpu_tensor_from_u32(gpu, offsets,
                                               (size_t)tiles + 1);
    if (!qt || !kt || !vt || !gt || !ot || !mt || !rt || !ft) return 0;
    double best_ms = INFINITY;
    for (int repeat = 0; repeat < repeats; repeat++) {
        if (repeat) {
            /* Restore the base output so every run adds once. */
            h3_gpu_tensor_free(ot);
            ot = h3_gpu_tensor_from_bf16(gpu, tmp, elements);
            if (!ot) return 0;
        }
        h3_gpu_stats before, after;
        h3_gpu_get_stats(gpu, &before);
        if (!h3_gpu_begin(gpu) ||
            !h3_gpu_vsa_coarse_bf16(gpu, ot, qt, kt, vt, gt, mt, rt, ft,
                                    rows, heads, dim, tiles, scale,
                                    qkv_head_major, output_head_major) ||
            !h3_gpu_submit(gpu)) {
            fprintf(stderr, "VSA test failed: %s\n", h3_gpu_error(gpu));
            return 0;
        }
        h3_gpu_get_stats(gpu, &after);
        double ms = (after.gpu_seconds - before.gpu_seconds) * 1000.0;
        if (ms < best_ms) best_ms = ms;
    }
    h3_gpu_tensor_read_bf16(ot, tmp, elements);
    if (output_head_major) from_head_major(actual, tmp, rows, heads, dim);
    else memcpy(actual, tmp, elements * sizeof(*actual));
    float max_error = 0.0f;
    for (size_t i = 0; i < elements; i++) {
        float got = from_bf16(actual[i]) - from_bf16(ob[i]);
        float difference = fabsf(got - expected[i]);
        if (difference > max_error) max_error = difference;
    }
    *max_error_out = max_error;
    *gpu_ms_out = best_ms;
    h3_gpu_tensor_free(qt); h3_gpu_tensor_free(kt); h3_gpu_tensor_free(vt);
    h3_gpu_tensor_free(gt); h3_gpu_tensor_free(ot); h3_gpu_tensor_free(mt);
    h3_gpu_tensor_free(rt); h3_gpu_tensor_free(ft);
    free(q); free(k); free(v); free(gate); free(expected);
    free(qb); free(kb); free(vb); free(gb); free(ob); free(tmp);
    free(actual); free(offsets); free(cursor); free(grouped);
    return 1;
}

int main(void) {
    char error[256];
    h3_gpu *gpu = h3_gpu_create("h3_shaders.metal", error, sizeof(error));
    if (!gpu) {
        fprintf(stderr, "VSA test setup failed: %s\n", error);
        return 1;
    }
    int failed = 0;
    unsigned map[] = {0, 0, 1, 1, 2, 2, 3, 3};
    for (int layout = 0; layout < 4; layout++) {
        float max_error;
        double ms;
        int qkv_hm = layout & 1, out_hm = (layout >> 1) & 1;
        if (!run_case(gpu, 8, 2, 4, map, 4, qkv_hm, out_hm, 0, 1,
                      &max_error, &ms)) return 1;
        if (!layout)
            printf("vsa coarse CPU/Metal max error: %.8g\n", max_error);
        else
            printf("vsa coarse CPU/Metal max error (qkv %s, output %s): "
                   "%.8g\n", qkv_hm ? "head-major" : "row-major",
                   out_hm ? "head-major" : "row-major", max_error);
        if (!(max_error < 0.02f)) failed = 1;
    }
    /* Near-real size from the FastH3 review (notes/h3-fasth3-review-1025.md
     * item 5): 5946 rows, 56 heads, head_dim 128, contiguous 64-row tiles
     * with a short edge tile. */
    const unsigned rows = 5946, heads = 56, dim = 128;
    const unsigned tiles = (rows + 63) / 64;
    unsigned *real_map = malloc(rows * sizeof(*real_map));
    if (!real_map) return 2;
    for (unsigned r = 0; r < rows; r++) real_map[r] = r / 64;
    float max_error;
    double ms;
    if (!run_case(gpu, rows, heads, dim, real_map, tiles, 1, 1, 1, 5,
                  &max_error, &ms)) return 1;
    printf("vsa coarse real-size (%u rows, %u heads, %u tiles) "
           "CPU/Metal max error: %.8g, GPU %.3f ms/layer\n",
           rows, heads, tiles, max_error, ms);
    if (!(max_error < 0.02f)) failed = 1;
    free(real_map);
    h3_gpu_free(gpu);
    return failed;
}
