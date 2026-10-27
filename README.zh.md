<h1 align="center">h3-metal</h1>

<p align="center">
  <strong>Mac 上最快的 MiniMax-H3 推理引擎——专为 Apple 芯片而生。</strong><br>
  热机比原版 h3.c 快约 <b>4 倍</b>，比原生 MLX 实现快 <b>7 倍多</b>。
</p>

<p align="center">
  <a href="#性能">性能</a> ·
  <a href="#快速开始">快速开始</a> ·
  <a href="#为什么快">为什么快</a> ·
  <a href="#路线图">路线图</a> ·
  <a href="README.md">English</a>
</p>

## 关于

**h3-metal** 是 MiniMax-H3 文生视频/音频在 Apple GPU 上的专用引擎：一个模型，一个引擎，直接照着
Metal 写。文本编码器、DiT 去噪、视频 VAE、音频 VAE 全在同一个原生二进制、同一条 Metal 命令流里跑
——提示词到 GPU 之间没有 Python，没有模型转换，每一步也不穿过框架边界。

它从 [antirez/h3.c](https://github.com/antirez/h3.c) 出发，为芯片继续往下压：FastH3 4 步蒸馏权重、
GPU 块稀疏注意力、常驻的解码器和 DiT、int8 Metal TensorOps 内核。

## 动态

- **2026-10-27：** FastH3 4 步权重 + 块稀疏细注意力默认开启。热机一条只要 **18.55 秒**——
  是原版 h3.c 的 **3.95 倍**、MLX 实现的 **7.3 倍**。

## 性能

<!-- latest-performance:start -->
测试机：**Apple M5 Max（40 核 GPU）、128 GB、macOS 27**。每一行条件相同：**768x448、49 帧、种子 0、
真权重**，一次只跑一个引擎。

- **冷启动** —— 新进程的第一条：权重加载、文本编码和准备工作都算在内。
- **热机** —— 同一进程里后面的条目，已加载的权重和准备好的状态被复用。

冷对冷、热对热比较。

| 引擎 | 步数 | 冷启动 | 热机 | 对 MLX（冷 / 热） | 对 h3.c（冷 / 热） |
| --- | --- | --- | --- | --- | --- |
| `PipeNetwork/minimax-h3-mlx`（4-bit） | 20 | 153.1 秒 | 134.6 秒 | 1.00× / 1.00× | 0.44× / 0.54× |
| `antirez/h3.c` 默认预设 | 20 | 67.6 秒 | 73.3 秒 | 2.26× / 1.84× | 1.00× / 1.00× |
| **h3-metal** 原版权重 | 20 | **63.3 秒** | **39.5 秒** | **2.42× / 3.41×** | **1.07× / 1.86×** |
| **h3-metal** + FastH3 + 稀疏注意力 | 4 ¹ | **48.23 秒** | **18.55 秒** | **3.17× / 7.3×** | **1.40× / 3.95×** |
<!-- latest-performance:end -->

¹ FastH3 是 4 步蒸馏的 DiT 权重
（[FastVideo/FastH3](https://huggingface.co/FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree)）；
块稀疏注意力在 GPU 上按粗注意力分数，为每个查询块从 120 个视频块里挑最相关的 12 个。分辨率、帧数、
种子和其他行一致。20 步各行的热机取第 2–4 条平均，FastH3 行取第 2 条。

用 [`bench/run_fox2.sh`](bench/run_fox2.sh) 可复现；各阶段耗时看 `--profile` 输出。

## 快速开始

需要 Apple 芯片的 macOS，把 MiniMax-H3 的 Hugging Face 快照放在 `./MiniMax-H3`（出 mp4 需要 `PATH` 里有 FFmpeg）。

```sh
git clone https://github.com/zuiho-kai/h3-metal && cd h3-metal
make -j8
./h3 --info -d ./MiniMax-H3
```

用原版权重生成：

```sh
./h3 -d ./MiniMax-H3 -p "黎明时分，一只红狐从长满苔藓的倒木上跃下" \
     --width 768 --height 448 --frames 49 --steps 20 --seed 0
```

用 FastH3 生成（把 FastH3 的 `transformer/` 目录放到 `./MiniMax-H3/FastH3-4step/transformer`）：

```sh
./h3 -d ./MiniMax-H3 --fasth3 -p "黎明时分，一只红狐从长满苔藓的倒木上跃下" \
     --width 768 --height 448 --frames 49 --seed 0
```

不带 `-p` 会进入交互会话，模型常驻，热机数字就是这样来的。完整命令手册（预设、参考输入、分辨率、
性能分析）见 [`docs/USAGE.md`](docs/USAGE.md)。

## 为什么快

**原生 Metal。** 整条流水线一个进程、一条 Metal 命令流。一步去噪只有几次提交，不是穿过框架的几百次派发。

**GPU 块稀疏注意力。** 粗分数挑出 120 个视频块里最相关的 12 个，细注意力只取这些块，变长、不补齐。
画面一致，热机从 20.32 秒降到 18.55 秒。

**4 步代替 20 步。** FastH3 蒸馏 DiT 直接载入引擎，张量布局映射到 h3.c 的内核上。

**int8 TensorOps 内核。** QKV、注意力输出、MLP 投影跑在 Metal 4 TensorOps 的动态对称 int8 上——
M5 Max 实测 119.3 TOPS。

**复用。** 文本条件、准备好的 DiT 和视频解码器跨条目常驻，下一条直接跳过加载和编码。

## 路线图

- **AI 自动搜索循环** —— 无人值守：模型提内核和调度补丁，冻结的测试台编译并测冷热，裁判决定晋级或淘汰，
  全部写进公开账本。引擎不靠人手调参也一直变快。
- **更快的冷启动** —— int8 权重落盘缓存，直接读进常驻 Metal 缓冲。
- **更省内存** —— 追上原版 h3.c 的 30 GB 峰值。
- **预编译二进制**和一行安装。

## 致谢与许可证

h3-metal 衍生自 Salvatore Sanfilippo 的 [antirez/h3.c](https://github.com/antirez/h3.c)，MIT 许可证，
见 [LICENSE](LICENSE)。部分 Metal int8 内核改编自 ccv 的 Metal FlashAttention（BSD-3-Clause），见
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。仓库不含模型权重，请从原发布方获取。
