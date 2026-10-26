# Qwen3.8-Flash-Next 部署细节（Strata + GSQ-RCO IQ3_XXS，单卡 16 GB）

> 2026-10-12 实际部署记录。`<尖括号>` 为需替换的值。

## 1. 模型

| 项目 | 值 |
|---|---|
| 模型 | Qwen3.8-Flash-Next（Qwen 官方，2026-08-26） |
| 结构 | MoE：主模型 125B 总参 / 每 token 激活 6B；另有 51B N-gram embedding 表 + 约 4B MTP 头 |
| 层 | 48 层：36 层 GDN（线性注意力）+ 12 层 QSA（只有这 12 层需要 KV cache） |
| 专家 | 每层 512 个 routed experts（48 × 512 = 24576），每 token 激活 10 个 + 1 个 shared |
| 原生上下文 | 262144 |
| 推理 | 必须思考的推理模型；模板档位 xhigh（默认）/ medium / low，另可关闭思考 |
| 许可 | 基座 qwen-community-1.0（HF 上量化版元数据写 Apache-2.0，正文说继承基座许可，商用前自行确认） |

## 2. 量化：ISTA-DASLab GSQ-RCO

- GSQ（arXiv 2604.18556）：Gumbel-Softmax 标量量化，负责每个张量怎么量化。
- RCO（arXiv 2605.00649）：在总大小预算下给每个张量分配量化类型。
- 仓库：`https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`；Coder 版另有 `...-GSQ-RCO-Coder-GGUF`。
- 每个档位两个分片：shard1 = transformer 权重；shard2 = N-gram 表（固定 IQ4_NL，28.8 GB，可放 SSD 用 mmap 读）。

| 档位 | bpw | shard1 | 总大小 | 任务均分（BF16 = 93.12） | 适合内存 |
|---|---|---|---|---|---|
| Coder IQ1_M（专家减半） | 有效 1.89 | 29.6 GB | 58.4 GB | 编码保留 91–99%，中文/通用下降 | 32 GB |
| Q2_0 | 2.40 | 37.6 GB | 66.4 GB | 89.07 | 48 GB |
| IQ2_XS | 2.50 | 39.2 GB | 68.0 GB | 89.16 | 48 GB |
| **IQ3_XXS（本次）** | 3.00 | 47.0 GB | 75.8 GB | **92.57** | 建议 60 GB，46 GB 实测可跑 |
| IQ3_S | 3.50 | 54.8 GB | 83.6 GB | 93.26 | 64 GB |

分数是 DASLab 自测（AIME25 / GPQA-D / LCB v6 平均），未独立复现。

## 3. 运行时：Strata

- `https://github.com/Niko1221/Strata`，MIT，本次固定 **v0.1.43**（提交 7695859，2026-10-11，当时最新）。
- 基于 llama.cpp / ggml 改造，专为 Flash-Next 做分层放置：
  - **显存**：dense 部分 + 最热的专家缓存 + MTP draft 层 + KV 窗口
  - **内存**：其余专家常驻并锁页（resident low-RAM 模式），adaptive swap 与 GPU 缓存交换
  - **SSD**：N-gram 表（shard2）mmap；experts pack 供长 prompt 借用槽位读取
- 自带接口：OpenAI `/v1/chat/completions`、Anthropic `/v1/messages`、网页聊天 `/`。

## 4. 硬件（本次实测）

| 项目 | 值 | 备注 |
|---|---|---|
| GPU | RTX 4080 SUPER 16 GB，PCIe 4.0 x16 | 官方最低 12 GB |
| 内存 | 46 GB DDR4，开 XMP 跑 3200 | 2133 → 3200 后 decode +24%，见 §9 |
| BIOS | Resizable BAR 开（BAR1 16 GB），CSM 关 | 实测 PCIe 带宽 11.0 → 14.0 GB/s |
| CPU | i7-12700，AVX2 | 必须 AVX2；有 AVX-512 时 setup 会多写一份约 40 GB 专家副本 |
| 磁盘 | ext4 NVMe，空闲 ≥ 160 GB | 实际占用见 §6 |
| 系统 | Ubuntu 24.04，驱动 595.99.02（Ubuntu 源 nvidia-driver-595-open） | |

## 5. 安装

### 5.1 CUDA Toolkit 13.0

本机编译引擎需要 nvcc。用 **13.0**（setup.py 提示早期 13.2 编出的引擎会输出乱码）。驱动不动。

```bash
curl -fsSL -o cuda-keyring.deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring.deb
# 防止以后 apt upgrade 用 NVIDIA 源替换 Ubuntu 驱动
sudo tee /etc/apt/preferences.d/99-keep-ubuntu-nvidia >/dev/null <<'EOF'
Package: nvidia-* libnvidia-* xserver-xorg-video-nvidia-* linux-modules-nvidia-* linux-objects-nvidia-* linux-signatures-nvidia-* libxnvctrl0 cuda-drivers*
Pin: release l=NVIDIA CUDA
Pin-Priority: -1
EOF
sudo apt-get update
sudo apt-get install -y cuda-toolkit-13-0 build-essential git python3-venv
```

### 5.2 Strata + 模型

```bash
git clone --branch v0.1.43 --depth 1 https://github.com/Niko1221/Strata.git ~/Strata
cd ~/Strata
export CUDA_HOME=/usr/local/cuda-13.0 PATH=/usr/local/cuda-13.0/bin:$PATH
nohup setsid ./setup.sh --yes --family qwen --model IQ3_XXS --build --cuda 13 --data-dir $HOME/Strata-data --context 8192 --kv int8 --low-ram auto --vision no --host 127.0.0.1 --no-browser --no-start > ~/strata-setup.log 2>&1 < /dev/null &
```

setup 依次做：编译引擎（138 个目标，几分钟）→ 下载两个分片（本次约 16 MB/s，约 80 分钟）→ 生成 low-RAM 用的 experts pack（42 GB）→ 生成 MTP draft 层 → 写出 `strata-iq3_xxs.json` 和 `run-iq3_xxs.sh`。`--model` 可选值：`Q2_0`、`IQ2_XS`、`IQ3_XXS`、`IQ3_S`，Coder 用 `--family coder`。

## 6. 文件布局（IQ3_XXS，共约 119 GB）

```
~/Strata/                              74 MB 源码 + engine/strata（编译产物）
~/Strata-data/models/IQ3_XXS/
    Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf   47.04 GB  transformer
    Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf   28.80 GB  N-gram 表
~/Strata-data/packs/iq3_xxs/           42 GB   experts pack + tokenizer
~/Strata-data/mtp/                     约 6.5 GB  rt/（787 MB，运行用）、tensors/、mtp-q2_0.gguf
```

## 7. 最终配置 `~/Strata/strata-iq3_xxs.json`

```json
{
 "exe": "/home/<USER>/Strata/engine/strata",
 "args": [
  "--pack", "/home/<USER>/Strata-data/packs/iq3_xxs",
  "--native", "/home/<USER>/Strata-data/models/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf",
  "--ple-gguf", "/home/<USER>/Strata-data/models/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf",
  "--expert-profile", "/home/<USER>/Strata/data/expert-profile.bin",
  "--expert-cache", "auto",
  "--prefill", "auto",
  "--spec", "4",
  "--spec-min-p", "0.5",
  "--mtp", "/home/<USER>/Strata-data/mtp/rt",
  "--max-context", "131072",
  "--resident-experts",
  "--kv-resident", "32768"
 ],
 "cwd": "/home/<USER>/Strata",
 "tokenizer": "/home/<USER>/Strata-data/packs/iq3_xxs/tokenizer",
 "model_name": "qwen3.8-flash-next-iq3_xxs",
 "log": "/home/<USER>/Strata/strata-iq3_xxs.log",
 "lib_dirs": ["/usr/local/cuda-13.0/bin", "/usr/local/cuda-13.0/lib64"],
 "port": 8080,
 "gpu": 0,
 "gpus_asked": true,
 "host": "<LISTEN_IP>",
 "open_browser": false,
 "env": {
  "STRATA_ROUTE_TAIL_SKIP": "0",
  "STRATA_RESIDENT_HEADROOM_GIB": "6"
 },
 "reasoning_budget_tokens": 4096,
 "reasoning_close_retry": true,
 "reasoning_effort": "medium",
 "fit_max_tokens": true,
 "sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 20},
 "expert_profile_save": "expert-profile-learned.bin"
}
```

API key 不写在这里，用环境变量 `STRATA_API_KEY` 传入（systemd 的 `EnvironmentFile`）。

### 相对 setup 默认值改了什么

| 配置 | setup 默认 | 改为 | 原因 |
|---|---|---|---|
| `--max-context` | 8192 | 131072 | KV 只在 12 层 QSA，int8 约 13.7 KB/token，128K ≈ 1.75 GB |
| `--kv-resident` | 无 | 32768 | KV streaming：完整 KV 放 3 GiB 锁页内存，显存只留 32K 窗口；读到的数值相同，无损 |
| `STRATA_ROUTE_TAIL_SKIP` | 7（v0.1.42 起 CUDA 默认开） | 0 | 有损近似：验证窗口里排名靠后的未缓存专家被跳过。关掉保质量 |
| `STRATA_RESIDENT_HEADROOM_GIB` | 4 | 6 | setup 对约 47 GB 内存的建议，给系统多留 2 GiB；测速无差别 |
| `reasoning_budget_tokens` | 无（只受 max_tokens 限制） | 4096 | 思考到 4096 token 自动收尾再作答 |
| `reasoning_close_retry` | false | true | 想完没正文时自动补一次作答 |
| `reasoning_effort` | 模型默认 | medium | 请求未指定档位时使用 |
| `fit_max_tokens` | false | true | prompt + max_tokens 超上下文时自动缩小 max_tokens，而不是返回 400 |
| `expert_profile_save` | 无 | `expert-profile-learned.bin` | 每 10 分钟（请求间隙）和正常退出时保存自适应缓存学到的常用专家，下次启动直接按它填 GPU 缓存；只改专家放哪里，不改计算 |
| `sampling` | 无（不带采样参数的请求一律 greedy） | temperature 1.0 / top_p 0.95 / top_k 20 | Qwen 官方思考模式推荐值。只对没带采样参数的请求生效；不设的话 greedy 写长文本会陷入重复（见 §11 第 9 条） |

保持默认：`--spec 4`、`--spec-min-p 0.5`（MTP 投机解码）、`--expert-cache auto`、`--prefill auto`、KV 精度 int8。

## 8. 启动后的资源分布（从启动日志读）

| 位置 | 内容 | 大小 |
|---|---|---|
| 显存 | GPU 专家缓存 | 约 5100–5280 槽，8.3–8.6 GiB |
| 显存 | MTP draft 层 | 809 MiB |
| 显存 | KV 窗口 + dense + 缓冲 | 合计占满到约 15.5 / 16 GB，剩约 370 MiB |
| 内存（锁页） | 常驻专家 | 约 32.6 GiB |
| 内存（锁页） | 完整 KV | 3.00 GiB |
| 系统可用 | — | 约 5 GiB |
| SSD | N-gram 表 mmap；长 prompt 时部分借用槽位读 experts pack | — |

关键日志行：

```
strata generate: KV streaming: 32768 of 131072 cells per QSA layer in VRAM, the K/V in 3.00 GiB of pinned RAM
strata generate: expert cache 5278 slots, 8.56 GiB of VRAM
strata generate: resident RAM mode: 32.58 GiB of experts in RAM (page-locked) ... (no file reads)
strata generate: PCIe probe: 14.0 GB/s host->device ... -> pcie_frac 0.39
[strata] thinking budget: 4096 tokens
[strata] thinking level: medium
```

必须给进程足够的锁页上限（systemd 里 `LimitMEMLOCK=infinity`），否则无法锁定约 36 GiB。首次加载约 35–70 秒，期间机器可能短暂卡顿。

## 9. 性能与调参结果

测速工具：Strata 自带 `tools/strata_bench.py`（thinking 关，temperature 0，每组 3–5 次取中位数）。

最终配置（128K，XMP 3200 + ReBAR）：

| 测试 | 结果 |
|---|---|
| 短对话 decode | 58.8 tok/s |
| agent 多轮 decode（约 14K，复用前缀缓存） | 72.2 tok/s |
| 4K prompt prefill | 1168 tok/s，首字 3.5 s |
| 16K prompt prefill | 1717 tok/s，首字 9.4 s |
| 89K 大海捞针 | 答对；prefill 1540 tok/s，首字约 58 s；该长度下 decode 38 tok/s |
| 111K prompt + max_tokens 32000 | 正常返回（fit_max_tokens 生效） |

以下为早期 46 GB 配置的历史测试。2026-10-26 的 64 GB 配置重新验证后采用了 overlap 与任务粒度调整，见 [新实测](benchmarks-2026-10-26.md)。

各项改动的实测影响：

| 改动 | decode | 结论 |
|---|---|---|
| 8K 基线（内存 2133） | 52.7 tok/s | — |
| `STRATA_ROUTE_TAIL_SKIP=7` | 67.7（+28%） | **有损**，未采用 |
| `--spec-min-p` 0.3 / 0.7 | 52.8 / 53.1 | 无差别 |
| `--spec 5` | 52.6 | 无差别 |
| `STRATA_RESIDENT_HEADROOM_GIB=4` | 52.3 | 无差别 |
| `STRATA_EXCHANGE_ROTATE=1` | 52.8 | IQ3_XXS 专家块大小不一，不生效 |
| `STRATA_ADAPT_OVERLAP=1` + `..._RESIDENT=1` | 51.1 | 略慢 |
| `--aux-cpus auto` | 52.3 | 无差别 |
| `--pool-workers 7` | 52.6 | 无差别 |
| CPU 调频 performance | 51.9 | 无差别 |
| 128K + `--kv-resident 32768` | 47.6（−10%） | 专家缓存少约 500 槽，换来 128K，采用 |
| 内存 XMP 3200 + ReBAR | 58.8（+24%） | 无损，**收益最大** |
| `STRATA_SPEC_COUPLED=1` + `STRATA_SPEC_GUMBEL=1`（temperature 1.0 测） | 68.8 → 69.2（+0.5%），草稿接受率 65.2% → 72.0% | 分布不变，但速度在波动内，未采用 |
| `--kv-grow` | — | 要求全部专家在内存，与低内存常驻模式冲突，46 GB 下不可用 |

结论：这个负载的瓶颈是内存带宽（CPU 侧专家计算和交换）。软件参数基本没有空间，硬件（内存频率、容量）才有用。

## 10. 思考控制

- 模型模板默认 **xhigh**；模板把 `high` 也映射成 xhigh；只认 xhigh / medium / low。
- Strata API 层档位：`reasoning_effort: none | low | medium | high`（`none` = 关闭思考）。
- Anthropic 格式：`thinking.budget_tokens` < 2K → low，< 8K → medium，其余 → high。
- 配置里的 `reasoning_effort` 只管**没指定档位**的请求；客户端（如 Claude Code 经网关转换后）通常会带 high，所以真正管用的是 `reasoning_budget_tokens` 硬上限。请求里自带 `reasoning_budget_tokens` 时以请求为准（`0` = 不限）。
- 不加上限时实测：45K prompt 的请求思考了 16959 token、耗时 273 秒。加 4096 上限后，同类难题 high 档约 100 秒、medium 档约 40 秒。

## 11. 已知限制和注意事项

1. **一次只处理一个请求**，其余在服务端排队（每 10 秒发 keep-alive）。`"parallel": N` 可开批处理，但官方说明在 12–16 GB 卡上每请求慢 10–25%，未采用。
2. **prompt + max_tokens 计入上下文**，不截断；不开 `fit_max_tokens` 时超限直接 400。
3. **长 prompt（50K+）首字慢**：46 GB 内存下约 64% 的借用槽位要从 SSD 读专家；110K 首字约 68 秒。客户端超时要设 ≥ 300 秒。
4. **输出不是逐字节可复现**：默认开启的短 prompt CPU prefill 分担（官方称平均 KL 约 0.004）和启动时自动测的 `pcie_frac` 会让每次重启后 temperature 0 的回答有细微差异。需要严格复现可设 `STRATA_PREFILL_CPU_SHARE=0` 并固定 `--pcie-frac`（未实测）。
5. **256K 在 46 GB 内存上不可行**：KV 需约 6 GiB 锁页内存，超出剩余可用量。需要 64 GB+。
6. **不要用 `--kv q4_0` / `k8v4` 省显存**：官方承认长文档困惑度变差（q4_0 +8–12%）。
7. 视觉未加载（`--vision no`）。需要图片输入要重新 setup 加 mmproj（0.91 GB），显存会更紧。
8. 混插内存跑 XMP 需充分测稳定性（memtester 1 轮只是快检，建议 MemTest86 过夜）；出现乱码或进程异常退出先怀疑内存。
9. **不带采样参数的长文本请求会陷入重复**：例如 pi 的自动压缩只传 max_tokens，Strata 无 `sampling` 块时按 greedy 解码，摘要每次写满输出上限后以 `length` 截断，pi 报 "generation hit the token cap"。特征：草稿接受率异常高（约 93%，正常约 70%）、decode 异常快。Strata 自带的循环检测只看思考部分，拦不住正文重复。加 §7 的 `sampling` 默认值后，同一请求 6769 token 自然结束（`stop`）。
10. pi 使用本模型时，摘要输出上限 = `min(0.8 × reserveTokens, maxTokens)`，默认只有 13107。在 `~/.pi/agent/settings.json` 加 `compaction.modelOverrides["<provider>/qwen3.8-flash-next"] = {"reserveTokens": 40960, "keepRecentTokens": 20000}`，并把 `models.json` 里该模型 `maxTokens` 设为 32768。pi 不热加载设置，改完要重启 pi。

## 12. 常用命令

```bash
sudo systemctl restart strata          # 改配置后重启
journalctl -u strata -f                # 服务输出（思考进度、done 行）
tail -f ~/Strata/strata-iq3_xxs.log    # 引擎日志（每个请求的 prompt/生成速度、草稿接受率）
grep -a 'strata serve: prompt' ~/Strata/strata-iq3_xxs.log | tail   # 最近请求的速度
cd ~/Strata && STRATA_KEY=<KEY> .venv/bin/python -I tools/strata_bench.py --url http://<LISTEN_IP>:8080 --runs 3   # 测速
```
