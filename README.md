<h1 align="center">h3-metal</h1>

<p align="center">
  <strong>Metal-native FastH3 on the Mac — 4-step distilled H3 with block-sparse attention, straight on Apple GPUs.</strong><br>
  FastH3 4-step renders a warm clip in <b>18.2 s</b>: about <b>4× faster</b> than upstream h3.c and <b>7× faster</b> than the MLX port (both 20-step).<br>
  Original H3 at 20 steps runs here too: <b>29.9 s</b> warm, <b>2.5×</b> h3.c and <b>4.5×</b> MLX.
</p>

<p align="center">
  <a href="#performance">Performance</a> ·
  <a href="#quickstart">Quickstart</a> ·
  <a href="#why-it-is-fast">Why It Is Fast</a> ·
  <a href="#roadmap">Roadmap</a> ·
  <a href="README.zh.md">中文</a>
</p>

## About

**h3-metal** runs FastH3 — the 4-step distilled MiniMax-H3 — natively on Metal, with the
original 20-step H3 alongside. It is a dedicated text-to-video/audio engine for Apple GPUs: one model, one
engine, written straight against Metal. Text encoder, DiT denoising, video VAE and audio VAE run
in one native binary as one Metal command stream — no Python between the prompt and the GPU, no
conversion step, no framework boundary per step.

It starts from [antirez/h3.c](https://github.com/antirez/h3.c) and pushes it further for the
chip: FastH3 4-step distilled weights, GPU block-sparse attention, a resident decoder and DiT,
and int8 Metal TensorOps kernels.

## News

- **2026-10-28:** Original H3 at 20 steps drops to **29.89 s** warm (from 39.5 s) —
  **2.45×** upstream h3.c and **4.50×** the MLX port.
- **2026-10-28:** h3-metal is now Metal-native FastH3: the 4-step distilled checkpoint with
  GPU block-sparse attention, written straight against Metal — no MLX, no Python.
- **2026-10-27 (later):** VAE projections skip the F32↔BF16 round trip and FC2 uses row-wise int8.
  A warm clip now takes **18.20 s** — **4.03×** upstream h3.c and **7.40×** the MLX port.
- **2026-10-27:** FastH3 4-step weights + block-sparse fine attention on by default.
  A warm clip now takes **18.55 s** — **3.95×** upstream h3.c and **7.3×** the MLX port.

## Performance

<!-- latest-performance:start -->
Test machine: **Apple M5 Max (40-core GPU), 128 GB, macOS 27**. Reference condition for every
row: **768x448, 49 frames, seed 0, real weights**, one engine at a time.

- **Cold** — the first clip in a fresh process: weight loading, text encoding and preparation
  are all inside the number.
- **Warm** — a later clip in the same process, where loaded weights and prepared state are reused.

Cold is compared with cold, warm with warm.

| Engine | Steps | Cold | Warm | vs. MLX (cold / warm) | vs. h3.c (cold / warm) |
| --- | --- | --- | --- | --- | --- |
| `PipeNetwork/minimax-h3-mlx` (4-bit) | 20 | 153.1 s | 134.6 s | 1.00× / 1.00× | 0.44× / 0.54× |
| `antirez/h3.c`, default preset | 20 | 67.6 s | 73.3 s | 2.26× / 1.84× | 1.00× / 1.00× |
| **h3-metal**, original H3 | 20 | **54.29 s** | **29.89 s** | **2.82× / 4.50×** | **1.25× / 2.45×** |
| **h3-metal**, FastH3 4-step + block-sparse attention | 4 ¹ | **48.23 s** | **18.20 s** | **3.17× / 7.40×** | **1.40× / 4.03×** |
<!-- latest-performance:end -->

¹ FastH3 is a 4-step distilled DiT checkpoint
([FastVideo/FastH3](https://huggingface.co/FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree));
block-sparse attention keeps the top 12 of 120 video tiles per query tile, picked on the GPU from
coarse scores. Same resolution, frames and seed as the other rows. The 20-step rows report warm
as the mean of clips 2–4; the FastH3 row reports warm as the mean of two A/B/B/A warm runs.

Reproduce a row with [`bench/run_fox2.sh`](bench/run_fox2.sh). Per-stage numbers come from
`--profile`.

## Quickstart

Requires macOS on Apple silicon and the MiniMax-H3 Hugging Face snapshot in `./MiniMax-H3`
(FFmpeg on `PATH` for mp4 output).

```sh
git clone https://github.com/zuiho-kai/h3-metal && cd h3-metal
make -j8
./h3 --info -d ./MiniMax-H3
```

Generate with the base weights:

```sh
./h3 -d ./MiniMax-H3 -p "a red fox leaps off a mossy log at dawn" \
     --width 768 --height 448 --frames 49 --steps 20 --seed 0
```

Generate with FastH3 (put the FastH3 `transformer/` folder at `./MiniMax-H3/FastH3-4step/transformer`):

```sh
./h3 -d ./MiniMax-H3 --fasth3 -p "a red fox leaps off a mossy log at dawn" \
     --width 768 --height 448 --frames 49 --seed 0
```

Without `-p` the same binary opens an interactive session that keeps the model resident, which is
where the warm numbers come from. The full CLI manual (presets, references, resolution, profiling)
is in [`docs/USAGE.md`](docs/USAGE.md).

## Why It Is Fast

**Metal, natively.** The whole pipeline is one process and one Metal command stream. A denoising
step is a handful of submits, not hundreds of dispatches through a framework.

**Block-sparse attention on the GPU.** Coarse tile scores pick the 12 most relevant of 120 video
tiles; the fine attention gathers only those tiles, variable length, no padding. Warm time drops
from 20.32 s to 18.55 s with the same picture.

**Four steps instead of twenty.** The FastH3 distilled DiT loads straight into the engine; its
tensor layout is mapped onto the h3.c kernels.

**int8 TensorOps kernels.** QKV, attention-out and MLP projections run on Metal 4 TensorOps with
dynamic symmetric int8 — 119.3 TOPS measured on M5 Max.

**Reused work.** Text conditioning, the prepared DiT and the video decoder stay resident across
clips, so the next clip skips loading and encoding.

## Roadmap

- **AI search loop** — an unattended loop where a model proposes kernel and scheduling patches,
  a frozen harness compiles and times each one cold and warm, and a judge promotes or rejects
  it into a public ledger. The engine keeps getting faster without hand-tuning.
- **Faster cold start** — int8 weight disk cache that loads straight into resident Metal buffers.
- **Leaner memory** — close the gap to upstream h3.c's 30 GB peak.
- **Prebuilt binaries** and a one-line installer.

## Credits and License

h3-metal is a derivative of [antirez/h3.c](https://github.com/antirez/h3.c) by Salvatore
Sanfilippo, released under the MIT License — see [LICENSE](LICENSE). Parts of the Metal int8
kernels are adapted from ccv's Metal FlashAttention (BSD-3-Clause) — see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Model weights are not included; get them from
their original publishers.
