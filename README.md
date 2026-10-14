# 【实测分享】16G 显卡 + 46G 内存本地跑 Qwen3.8-Flash-Next（125B MoE）：Strata + GSQ-RCO 部署全流程（Ubuntu 24.04）

> **一句话**：RTX 4080 SUPER 16G + 46G DDR4，Ubuntu 24.04，GSQ-RCO IQ3_XXS 量化，128K 上下文，短对话约 59 tok/s，agent 多轮约 72 tok/s，全程本地推理。
>
> 文中性能数字除特别注明外，均为我自己机器实测；不同 CPU、内存、显卡结果会有差异，仅供参考。

---

## 一、方案简介

| 组件 | 说明 |
|---|---|
| 模型 | **Qwen3.8-Flash-Next**：125B 总参 / 每 token 激活 6B 的 MoE，另有 51B N-gram embedding 表和约 4B 的 MTP 头；48 层里只有 12 层（QSA）需要 KV cache；原生 262K 上下文 |
| 量化 | **ISTA-DASLab GSQ-RCO GGUF**：GSQ 负责每个张量怎么量化，RCO 在总大小预算下给每个张量分配量化类型 |
| 运行时 | **[Strata](https://github.com/Niko1221/Strata)**（MIT）：常用专家放显存，其余专家锁在内存，N-gram 表放 SSD 用 mmap 读；自带 OpenAI / Anthropic 兼容接口和网页聊天 |

简单理解：显卡只放“灶台边常用的”，内存放“仓库里全部的”，SSD 放“查找表”。所以 16G 显卡也能跑 125B 的 MoE。

---

## 二、硬件要求与量化怎么选

**最低要求**：NVIDIA 12G 起（RTX 20–50 系列）；内存 32G 起；CPU 必须支持 AVX2；Ubuntu 原生系统（WSL 下 KV streaming 不可用）；空闲磁盘见下表，建议放 ext4 的 NVMe。

| 档位 | 下载大小 | 任务均分（BF16 = 93.12） | 生成速度（作者 RTX 5070 + 64G） | 适合内存 |
|---|---|---|---|---|
| Coder（专家减半） | 58.4 GB | 编码保留 91–99%，中文 / 通用会下降 | 55 tok/s | 32G |
| Q2_0 | 66.4 GB | 89.07 | 94 tok/s | 48G |
| IQ2_XS | 68.0 GB | 89.16 | 79 tok/s | 48G |
| **IQ3_XXS（本文）** | 75.8 GB | **92.57** | 62 tok/s | 官方建议 60G，46G 实测可用 |
| IQ3_S | 83.6 GB | 93.26 | 53 tok/s | 64G |

> 均分来自 DASLab 模型卡自测（AIME25 / GPQA-D / LCB v6 平均），速度来自 Strata 作者 README，都不是我复现的。

磁盘占用（IQ3_XXS）：两个分片 75.8G + low-RAM 模式的 experts pack 42G + MTP 约 6.5G + CUDA Toolkit 约 10G，**合计预留 160G 以上**。CPU 有 AVX-512 时 setup 还会多写一份约 40G 的专家副本。

---

## 三、我的实测

**机器**：RTX 4080 SUPER 16G（PCIe 4.0 x16）/ i7-12700 / 46G DDR4-3200（开 XMP）/ Resizable BAR 开 / Ubuntu 24.04 / 驱动 595（Ubuntu 源）

| 测试 | 结果 |
|---|---|
| 短对话 decode | **58.8 tok/s** |
| agent 多轮 decode（约 14K，复用前缀缓存） | **72.2 tok/s** |
| 4K prompt prefill | 1168 tok/s，首字 3.5 s |
| 16K prompt prefill | 1717 tok/s，首字 9.4 s |
| 111K prompt | 正常返回，首字约 68 s |
| 89K 大海捞针（开 XMP 前测） | 答对，prefill 1540 tok/s |

测速工具是 Strata 自带的 `tools/strata_bench.py`（temperature 0，关思考，每组 3–5 次取中位数）。

<details>
<summary>调参实测（哪些有用、哪些没用）</summary>


| 改动 | decode 变化 | 结论 |
|---|---|---|
| BIOS 开 XMP（2133 → 3200）+ ReBAR | **+24%** | 无损，收益最大 |
| `STRATA_ROUTE_TAIL_SKIP=7`（v0.1.42 起 CUDA 默认开） | +28% | **有损**，输出会变，我关掉了 |
| 128K + KV streaming | −10% | 换来 128K 上下文，采用 |
| `--spec-min-p` 0.3 / 0.7、`--spec 5` | 无差别 | 保持默认 |
| `STRATA_EXCHANGE_ROTATE=1` | 无差别 | IQ3_XXS 专家块大小不一，不生效 |
| `STRATA_ADAPT_OVERLAP=1` | 略慢 | 不用 |
| `--aux-cpus auto`、`--pool-workers 7`、CPU performance 调频 | 无差别 | 不用 |
| `STRATA_SPEC_COUPLED=1` + `STRATA_SPEC_GUMBEL=1` | +0.5%（草稿接受率 65% → 72%） | 在波动内，不用 |
| `--kv-grow` | — | 要求全部专家在内存，和低内存常驻模式冲突 |

结论：这个负载的瓶颈是**内存带宽和容量**（CPU 侧要算 GPU 放不下的专家）。软件参数基本没空间，内存频率和容量才有用。

</details>

---

## 四、部署步骤

### 4.1 BIOS（强烈建议）

以微星 B660 为例，其他主板名称类似：

1. **OC**：Extreme Memory Profile (X.M.P) 选 Profile 1；12 代 + DDR4 选 Gear 1；不要手动加压。
2. **Settings → Advanced → Windows OS Configuration**：BIOS UEFI/CSM Mode 选 **UEFI**（系统必须是 UEFI 安装，否则关 CSM 后进不去）。
3. **Settings → Advanced → PCI Subsystem Settings**：Above 4G memory 和 Re-Size BAR Support 都设为 **Enabled**。
4. F10 保存。第一次开机做内存训练可能黑屏 1–2 分钟，正常。

验证：

```bash
sudo dmidecode -t memory | grep "Configured Memory Speed"   # 应为 3200 MT/s
nvidia-smi -q | grep -A2 "BAR1 Memory"                     # Total 应为 16384 MiB（未开时 256 MiB）
sudo apt-get install -y memtester
sudo memtester 16G 1                                        # 必须 0 个 FAILURE
```

> 混插不同品牌内存条时 3200 不一定稳，不稳就降到 3000。memtester 跑 1 轮只是快检，建议再用 MemTest86 过夜。不稳定的内存会让模型悄悄输出乱码。

### 4.2 驱动 + CUDA Toolkit 13.0

驱动用 Ubuntu 官方源的 `nvidia-driver-595-open`（或你现有的驱动，不用动）。编译引擎需要 nvcc，**用 CUDA 13.0**：Strata 的 setup 提示早期 13.2 编出的引擎可能输出乱码。

```bash
curl -fsSL -o cuda-keyring.deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring.deb
```

**重要**：NVIDIA 源默认优先级是 600，以后执行 `apt upgrade` 可能把 Ubuntu 的驱动换成 NVIDIA 源的。先锁住驱动相关包：

```bash
sudo tee /etc/apt/preferences.d/99-keep-ubuntu-nvidia >/dev/null <<'EOF'
Package: nvidia-* libnvidia-* xserver-xorg-video-nvidia-* linux-modules-nvidia-* linux-objects-nvidia-* linux-signatures-nvidia-* libxnvctrl0 cuda-drivers*
Pin: release l=NVIDIA CUDA
Pin-Priority: -1
EOF
```

先模拟，确认不会删包、不会动驱动和内核，再安装：

```bash
sudo apt-get update
LC_ALL=C sudo apt-get -s install cuda-toolkit-13-0 | grep -E '^Remv|nvidia-driver|dkms|linux-image|linux-headers'   # 应该没有输出
sudo apt-get install -y cuda-toolkit-13-0 build-essential git python3-venv
/usr/local/cuda-13.0/bin/nvcc --version   # V13.0.x
```

> 只装 `cuda-toolkit-13-0`，**不要**装 `cuda` 或 `cuda-drivers` 元包，那两个会带驱动。

### 4.3 安装 Strata 并下载模型

固定版本，方便复现（我用的是 v0.1.43）：

```bash
git clone --branch v0.1.43 --depth 1 https://github.com/Niko1221/Strata.git ~/Strata
cd ~/Strata
export CUDA_HOME=/usr/local/cuda-13.0 PATH=/usr/local/cuda-13.0/bin:$PATH
```

无交互安装（只装不启动，下载时间长，建议放后台）：

```bash
nohup setsid ./setup.sh --yes --family qwen --model IQ3_XXS --build --cuda 13 --data-dir $HOME/Strata-data --context 8192 --kv int8 --low-ram auto --vision no --host 127.0.0.1 --no-browser --no-start > ~/strata-setup.log 2>&1 < /dev/null &
tail -f ~/strata-setup.log
```

- `--model` 可选 `Q2_0`、`IQ2_XS`、`IQ3_XXS`、`IQ3_S`；Coder 版用 `--family coder`。
- 过程：编译引擎（几分钟）→ 下载两个分片 → 生成 low-RAM 用的 experts pack（42G）→ 生成 MTP 草稿层 → 写出 `strata-iq3_xxs.json` 和 `run-iq3_xxs.sh`。
- 本机已有 nvcc 时 setup 不会再用 sudo 装 Toolkit；它不会改驱动、内核、`/etc/security/limits` 或 sysctl（我按 v0.1.43 源码核对过）。

### 4.4 修改配置

setup 默认的上下文只有 8K，并且默认开着有损的 tail-skip。用下面的脚本改，**改之前先备份**：

```bash
cd ~/Strata
cp strata-iq3_xxs.json strata-iq3_xxs.json.orig
python3 - <<'EOF'
import json
p = "strata-iq3_xxs.json"
c = json.load(open(p))
a = c["args"]
def setarg(k, v):
    if k in a:
        a[a.index(k) + 1] = v
    else:
        a.extend([k, v])
setarg("--max-context", "131072")
setarg("--kv-resident", "32768")
c.setdefault("env", {}).update({
    "STRATA_ROUTE_TAIL_SKIP": "0",
    "STRATA_RESIDENT_HEADROOM_GIB": "6",
})
c.update({
    "reasoning_budget_tokens": 4096,
    "reasoning_close_retry": True,
    "reasoning_effort": "medium",
    "fit_max_tokens": True,
    "sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 20},
    "expert_profile_save": "expert-profile-learned.bin",
})
json.dump(c, open(p, "w"), indent=1)
print("ok")
EOF
```

<details>
<summary>每一项改了什么、为什么</summary>


| 配置 | 默认 | 改为 | 原因 |
|---|---|---|---|
| `--max-context` | 8192 | 131072 | KV 只在 12 层 QSA，int8 约 13.7 KB/token，128K 约 1.75 GB |
| `--kv-resident` | 无 | 32768 | KV streaming：完整 KV 放 3 GiB 锁页内存，显存只留 32K 窗口；读到的数值完全相同，无损 |
| `STRATA_ROUTE_TAIL_SKIP` | 7 | 0 | v0.1.42 起 CUDA 默认开的有损近似：验证窗口里排名靠后的未缓存专家直接跳过。质量优先就关掉 |
| `STRATA_RESIDENT_HEADROOM_GIB` | 4 | 6 | setup 针对约 47G 内存的建议，给系统多留 2 GiB；测速无差别 |
| `reasoning_budget_tokens` | 无 | 4096 | 思考硬上限。模型模板默认 xhigh，不设的话我遇到过一次思考 1.7 万 token、4 分半 |
| `reasoning_close_retry` | false | true | 想完了没出正文时自动补一次作答，防止 agent 卡住 |
| `reasoning_effort` | 模型默认 | medium | 只对**没指定档位**的请求生效 |
| `fit_max_tokens` | false | true | prompt + max_tokens 超上下文时自动缩小 max_tokens，而不是直接 400 |
| `sampling` | 无 | 1.0 / 0.95 / 20 | Qwen 官方思考模式推荐值。不设的话，不带采样参数的请求一律 greedy，写长文本会陷入重复 |
| `expert_profile_save` | 无 | 文件名 | 每 10 分钟和正常退出时保存你常用的专家，下次启动直接按它填 GPU 缓存 |

内存只有 32–40G 的，128K 和 `--kv-resident` 要按自己的余量重新评估，可以先保持 8K–32K。

</details>

### 4.5 API key + systemd 开机自启

生成 key（不写进配置文件，也不出现在命令行参数里）：

```bash
sudo install -d -m 700 /etc/strata
echo "STRATA_API_KEY=strata-$(openssl rand -hex 24)" | sudo tee /etc/strata/api.env >/dev/null
sudo chmod 600 /etc/strata/api.env
```

新建 `/etc/systemd/system/strata.service`（把 `<你的用户名>` 换掉）：

```ini
[Unit]
Description=Strata (Qwen3.8-Flash-Next IQ3_XXS)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=<你的用户名>
Group=<你的用户名>
WorkingDirectory=/home/<你的用户名>/Strata
ExecStart=/home/<你的用户名>/Strata/run-iq3_xxs.sh
EnvironmentFile=/etc/strata/api.env
Restart=on-failure
RestartSec=15
TimeoutStartSec=600
TimeoutStopSec=60
LimitMEMLOCK=infinity
LimitNOFILE=65536

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now strata
journalctl -u strata -f    # 看到 ready: 就是起来了
```

> `LimitMEMLOCK=infinity` **必须有**：启动时要锁定约 33 GiB 专家和 3 GiB KV。首次加载 35–70 秒，期间机器可能短暂卡顿，属正常。

### 4.6 验证

```bash
K=$(sudo sed -n 's/^STRATA_API_KEY=//p' /etc/strata/api.env)
curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8080/v1/models; echo                         # 401：鉴权生效
curl -s -o /dev/null -w '%{http_code}' -H "Authorization: Bearer $K" http://127.0.0.1:8080/v1/models; echo   # 200
curl -s http://127.0.0.1:8080/v1/chat/completions -H "Authorization: Bearer $K" -H 'Content-Type: application/json' -d '{"model":"qwen3.8-flash-next","messages":[{"role":"user","content":"你好"}],"max_tokens":1024}'
```

启动日志里应该能看到这几行：

```
strata generate: KV streaming: 32768 of 131072 cells per QSA layer in VRAM, the K/V in 3.00 GiB of pinned RAM
strata generate: resident RAM mode: ... GiB of experts in RAM (page-locked) ... (no file reads)
[strata] sampling defaults from the config: temperature=1.0, top_p=0.95, top_k=20
[strata] thinking budget: 4096 tokens
```

测速：

```bash
cd ~/Strata && STRATA_KEY=$K .venv/bin/python -I tools/strata_bench.py --url http://127.0.0.1:8080 --runs 3
```

---

## 五、客户端接入

| 项目 | 填写 |
|---|---|
| Base URL | `http://127.0.0.1:8080/v1`（Anthropic 协议客户端填 `http://127.0.0.1:8080`） |
| API Key | `/etc/strata/api.env` 里的值 |
| 模型名 | 任意（Strata 只加载一个模型） |
| 上下文 | 131072 |
| 最大输出 | 16384（模型上限 32768） |
| 输入 | 仅文本（本文没加载视觉模块） |
| 能力 | 工具调用、流式、推理（`reasoning_content`） |
| 推理等级 | none / low / medium / high；high 会被模板升成 xhigh；全部受 4096 上限约束 |
| 超时 | 建议 ≥ 300 秒（长 prompt 首字慢） |

- **Claude Code**：Strata 自带 `/v1/messages`，`ANTHROPIC_BASE_URL` 指向 `http://127.0.0.1:8080` 即可。我自己是经网关中转用的，本机直连这一步没单独测过。
- **pi**：自动压缩默认只给摘要 13107 token，用这个模型会被截断报 "generation hit the token cap"。在 `~/.pi/agent/settings.json` 加 `compaction.modelOverrides["<provider>/<模型名>"] = {"reserveTokens": 40960, "keepRecentTokens": 20000}`，并把 `models.json` 里该模型 `maxTokens` 设为 32768，改完重启 pi。
- **对外提供**：不要直接把 8080 暴露到公网。需要远程用的话，走 Tailscale / WireGuard 内网，或者前面加带鉴权的网关。

---

## 六、踩坑汇总

1. **CUDA 用 13.0**，不要用早期 13.2，否则可能乱码。
2. **加了 NVIDIA 源一定要 pin 驱动包**，否则哪天 `apt upgrade` 就把驱动换了。
3. 系统是中文 locale 时 apt 输出是中文，脚本里 grep 英文关键字前加 `LC_ALL=C`。
4. **tail-skip 默认是开的**，而且是有损的。质量优先必须显式设 `STRATA_ROUTE_TAIL_SKIP=0`。
5. **模型默认 xhigh 思考**，客户端传 high 也会变成 xhigh。真正管得住的只有 `reasoning_budget_tokens`；配置里的 `reasoning_effort` 只管没指定档位的请求。
6. **max_tokens 算进上下文**，Strata 不截断，超了直接 400，所以开 `fit_max_tokens`。
7. **不带采样参数的请求默认 greedy**，长摘要会陷入重复（特征：草稿接受率异常高，约 90%，每次都写满上限），所以配 `sampling` 默认值。
8. **一次只处理一个请求**，其他请求在服务端排队（每 10 秒发 keep-alive）。`"parallel": N` 在 12–16G 卡上每个请求会慢 10–25%，我没开。
9. 46G 内存下，50K 以上的长 prompt 有一部分专家要从 SSD 读，prefill 会变慢，属正常。
10. 输出不是逐字节可复现：默认开启的短 prompt CPU prefill 分担和启动时自动测的 `pcie_frac` 会带来舍入级差异。
11. **256K 在 46G 内存下不可行**（KV 要约 6 GiB 锁页内存），想上 256K 得 64G 以上。
12. 不要用 `--kv q4_0` / `k8v4` 省显存，官方承认长文档困惑度会变差。
13. 改配置后 `sudo systemctl restart strata`，约 35 秒恢复；写脚本时注意 `pgrep -f` 会匹配到自己的命令行。

---

## 七、常用命令

```bash
sudo systemctl status strata          # 状态
sudo systemctl restart strata         # 改配置后重启
sudo systemctl stop strata            # 停止，释放约 40G 内存和显存
journalctl -u strata -f               # 服务输出（思考进度、done 行）
tail -f ~/Strata/strata-iq3_xxs.log   # 引擎日志（每个请求的速度、草稿接受率）
```

升级：`git fetch --tags && git checkout <新版本> && ./setup.sh`，**先看 release 说明里有没有新增默认开启的近似项**（tail-skip 就是 0.1.42 加的），setup 重跑会改写配置，记得把 4.4 的改动再加一遍。

---

## 八、参考链接

更细的部署记录见 [docs/deploy-details.md](docs/deploy-details.md)；带网关、Tailscale、客户端的全链路版见 [docs/full-chain.md](docs/full-chain.md)。


- Strata：https://github.com/Niko1221/Strata
- GSQ-RCO 量化模型：https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF
- GSQ 论文：https://arxiv.org/abs/2604.18556
- RCO 论文：https://arxiv.org/abs/2605.00649
- Qwen3.8-Flash-Next：https://github.com/QwenLM/Qwen3.8-Flash-Next
- Mac 用户可以看看 strata-mlx（非官方，Apple Silicon）：https://github.com/yibie/strata-mlx

---

## 九、说明

- 本文是个人部署记录，和 Strata、Qwen、ISTA-DASLab 均无关系。
- 许可：Strata 为 MIT；量化权重在 HF 元数据里标的是 Apache-2.0，但模型卡正文说继承基座许可（qwen-community-1.0），**商用前请自行确认**。
- 开 XMP、改 BIOS 有一定风险，操作前确认自己会清 CMOS。
- 有问题欢迎提 Issue，数据有出入也请指出。
