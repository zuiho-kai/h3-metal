# FastH3 tensor key map

Sources:

- FastH3 index: `FastH3-4step/transformer/diffusion_pytorch_model.safetensors.index.json`
- h3.c index: `MiniMax-H3/FL2VA/transformer/model.safetensors.index.json`
- FastH3 shard 1 header was read locally. Other shard headers were requested with HTTP `Range`; two requests were unavailable while the background download held the proxy. Repeated-layer schemas were inferred from shard 1 and the index.

## Counts

- One-to-one renames: **480**
- Split/concatenate candidates: **52** h3.c QKV tensors to **156** FastH3 Q/K/V tensors
- h3.c-only keys after the proposed map: **3**
- FastH3-only keys after the proposed map: **52**
- Net FastH3 surplus: **153** (`156 - 52 - 3`)
- Shape mismatches among mapped entries: **0**

## One-to-one renames

Examples:

- `blocks.0.norm1.weight` -> `transformer_blocks.0.norm1.weight`
- `blocks.0.attn.out_proj.weight` -> `transformer_blocks.0.attn.to_out.0.weight`
- `time_embedder.proj_in.weight` -> `time_embedder.linear_1.weight`

The map also covers the two token-refiner blocks and the top-level patch,
condition, audio, output, and time-embedding names.

## Split or concatenate

Examples:

- `blocks.0.attn.qkv_proj.weight` -> `transformer_blocks.0.attn.to_q.weight` + `to_k.weight` + `to_v.weight`
- `blocks.1.attn.qkv_proj.weight` -> `transformer_blocks.1.attn.to_q.weight` + `to_k.weight` + `to_v.weight`
- `token_refiner.blocks.0.attn.qkv_proj.weight` -> `token_refiner.refiner_blocks.0.attn.to_q.weight` + `to_k.weight` + `to_v.weight`

The three FastH3 matrices concatenate along dimension 0 to the h3.c QKV
shape (`21504 x 5376`); no shape mismatch was found.

## Unmatched modules

h3.c-only:

- `final_layer.adaln_proj.linear.weight` and `.bias` (2): final AdaLN parameters
- `rope.inv_freq` (1): static RoPE frequency tensor

FastH3-only:

- `transformer_blocks.*.attn.to_gate_compress.weight` (50): VSA-H3 gate-compression projection, one per transformer block
- `norm_out.linear.weight` and `.bias` (2): FastH3 output normalization projection

Thus the surplus 153 keys are not refiner layers: the refiner's 21 keys are
covered by one-to-one renames plus its two QKV splits. The net surplus is
caused by the 104 extra split tensors, 52 VSA gate-compression tensors, and
the two FastH3 output-linear tensors, less the three h3.c-only tensors.
