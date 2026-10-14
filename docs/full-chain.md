# 单卡 16 GB 显卡私有部署 Qwen3.8-Flash-Next（Strata）完整方案

> 基于 2026-10-12 实际部署整理。`<尖括号>` 是需要换成自己的值。**任何 key 都不要写进文档或聊天记录。**

## 0. 架构

```
客户端（Claude Code / ZCode / pi / 任意 OpenAI 兼容客户端）
   │  HTTPS  https://<NEWAPI_DOMAIN>/v1     ← new-api 令牌
   ▼
VPS：nginx → new-api（分组 local-gpu，OpenAI 类型渠道，并发 3）
   │  Tailscale（WireGuard 加密，不暴露公网端口）
   ▼
GPU 主机：Strata 服务  http://<TAILNET_IP>:8080   ← Strata 自己的 API key
   └─ Qwen3.8-Flash-Next GSQ-RCO IQ3_XXS，128K 上下文，systemd 开机自启
```

只在 GPU 主机本机使用的话，§6 的 Tailscale 和 §7 的 new-api 可以跳过，Strata 监听 `127.0.0.1` 即可。

---

## 1. 总清单（按顺序打勾）

- [ ] §2 硬件满足要求，选好量化档位
- [ ] §3 BIOS：XMP + Resizable BAR，memtester 通过
- [ ] §4 Ubuntu 24.04 + Ubuntu 源 NVIDIA 驱动；装 CUDA Toolkit 13.0，并锁住驱动不被 NVIDIA 源替换
- [ ] §5 安装 Strata v0.1.43，下载模型、编译引擎
- [ ] §6 改 Strata 配置：128K、关有损优化、思考上限、fit_max_tokens
- [ ] §7 Tailscale + API key + systemd 服务
- [ ] §8 new-api：分组、渠道、定价、令牌
- [ ] §9 客户端配置
- [ ] §10 验收

---

## 2. 硬件与量化选择

| 项目 | 本次实际 | 最低 / 建议 |
|---|---|---|
| GPU | RTX 4080 SUPER 16 GB（Ada） | NVIDIA 12 GB 起；RTX 20–50 系列 |
| 内存 | 46 GB DDR4（2×8 + 2×16） | 32 GB 起；64 GB 能跑满所有量化 |
| CPU | i7-12700（AVX2，无 AVX-512） | 必须支持 AVX2 |
| 磁盘 | ext4 根分区 200 GB | 空闲 ≥ 160 GB（权重 75.8 GB + experts.bin 42.9 GB + MTP 约 6 GB + CUDA 约 10 GB） |
| 系统 | Ubuntu 24.04.5，内核 7.0，驱动 595.99.02 | Ubuntu 24.04 原生（WSL 不能做 KV streaming） |

量化档位（任务均分以 BF16 = 93.12 为基准；速度为作者 RTX 5070 + 64 GB 数据）：

| 档位 | 下载 | 均分 | 速度 | 适合内存 |
|---|---|---|---|---|
| Coder（专家减半） | 58.4 GB | 编码保留 91–99%，中文/通用下降 | 55 tok/s | 32 GB |
| Q2_0 | 66.4 GB | 89.07 | 94 tok/s | 48 GB |
| IQ2_XS | 68.0 GB | 89.16 | 79 tok/s | 48 GB |
| **IQ3_XXS（本次选用）** | 75.8 GB | **92.57** | 62 tok/s | 建议 60 GB，46 GB 实测可用 |
| IQ3_S | 83.6 GB | 93.26 | 53 tok/s | 64 GB |

---

## 3. BIOS（微星 B660 为例，其他板子名称类似）

1. 开机按 Del，F7 进高级模式。
2. OC：Extreme Memory Profile (X.M.P) → Profile 1（DDR4-3200）；Gear 模式 Gear 1；不要手动加压。
3. Settings → Advanced → Windows OS Configuration：BIOS UEFI/CSM Mode → **UEFI**（系统必须是 UEFI 安装）。
4. Settings → Advanced → PCI Subsystem Settings：Above 4G memory → Enabled；Re-Size BAR Support → Enabled。
5. F10 保存。首次开机内存训练可能黑屏 1–2 分钟，属正常。

验证：

```bash
sudo dmidecode -t memory | grep "Configured Memory Speed"     # 应为 3200 MT/s
nvidia-smi -q | grep -A2 "BAR1 Memory"                       # Total 应为 16384 MiB（之前 256 MiB）
sudo apt-get install -y memtester && sudo memtester 16G 1    # 先停掉 Strata 再测，必须 0 FAILURE
```

混插内存条在 3200 下不稳时降到 3000。建议再用 MemTest86 U 盘跑一整晚。本项实测：decode +24%，PCIe 带宽 11.0 → 14.0 GB/s，**无损**。

---

## 4. 系统准备：驱动 + CUDA Toolkit 13.0

驱动用 Ubuntu 官方源的 `nvidia-driver-595-open`。CUDA Toolkit 用 NVIDIA 源的 **13.0**（Strata 的 setup 提示早期 13.2 编译出的引擎会输出乱码）。

```bash
curl -fsSL -o cuda-keyring.deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring.deb

# 关键：NVIDIA 源默认优先级 600，以后 apt upgrade 可能把驱动换掉。禁止从 NVIDIA 源装驱动相关包：
sudo tee /etc/apt/preferences.d/99-keep-ubuntu-nvidia >/dev/null <<'EOF'
Package: nvidia-* libnvidia-* xserver-xorg-video-nvidia-* linux-modules-nvidia-* linux-objects-nvidia-* linux-signatures-nvidia-* libxnvctrl0 cuda-drivers*
Pin: release l=NVIDIA CUDA
Pin-Priority: -1
EOF

sudo apt-get update
# 先模拟：下面这条必须没有输出（不能有删除、驱动、DKMS、内核相关的包）
LC_ALL=C sudo apt-get -s install cuda-toolkit-13-0 | grep -E '^Remv|nvidia-driver|dkms|linux-image|linux-headers'
sudo apt-get install -y cuda-toolkit-13-0     # 只装工具链，不要装 cuda 或 cuda-drivers 元包
/usr/local/cuda-13.0/bin/nvcc --version       # V13.0.x
sudo apt-get install -y build-essential git python3-venv
```

---

## 5. 安装 Strata

```bash
git clone --branch v0.1.43 --depth 1 https://github.com/Niko1221/Strata.git ~/Strata
cd ~/Strata
export CUDA_HOME=/usr/local/cuda-13.0 PATH=/usr/local/cuda-13.0/bin:$PATH
nohup setsid ./setup.sh --yes --family qwen --model IQ3_XXS --build --cuda 13 --data-dir $HOME/Strata-data --context 8192 --kv int8 --low-ram auto --vision no --host 127.0.0.1 --no-browser --no-start > ~/strata-setup.log 2>&1 < /dev/null &
tail -f ~/strata-setup.log
```

- 过程：编译引擎（几分钟）→ 下载两个分片（本次约 16 MB/s，约 80 分钟）→ 生成 low-RAM 用的 experts pack（42 GB）→ 生成 MTP draft 层 → 写出 `strata-iq3_xxs.json` 和 `run-iq3_xxs.sh`。
- 本机已有 nvcc 时 setup 不会再调用 sudo 装 Toolkit。setup 不会修改驱动、内核、`/etc/security/limits`、sysctl。
- 先只安装（`--no-start`），改完配置再启动。

---

## 6. 调整 Strata 配置（`~/Strata/strata-iq3_xxs.json`）

改之前先 `cp strata-iq3_xxs.json strata-iq3_xxs.json.orig`。最终相对 setup 默认值的改动：

`args` 里：

```json
"--max-context", "131072",
"--kv-resident", "32768"
```

顶层（`args` 之外）：

```json
"host": "<TAILNET_IP>",
"env": {
  "STRATA_ROUTE_TAIL_SKIP": "0",
  "STRATA_RESIDENT_HEADROOM_GIB": "6"
},
"reasoning_budget_tokens": 4096,
"reasoning_close_retry": true,
"reasoning_effort": "medium",
"fit_max_tokens": true
```

| 配置 | 作用 | 为什么 |
|---|---|---|
| `--max-context 131072` + `--kv-resident 32768` | 128K 上下文；KV 放 3 GiB 锁页内存，显存只留 32K 窗口 | 无损；46 GB 内存下 128K 是上限，256K 需要 64 GB+ |
| `STRATA_ROUTE_TAIL_SKIP=0` | 关闭 v0.1.42 起默认开启的有损跳专家 | 质量优先。打开约 +28% decode，但输出会变 |
| `STRATA_RESIDENT_HEADROOM_GIB=6` | 给系统多留 2 GiB 内存 | setup 针对约 47 GB 内存的建议，测速无差别 |
| `reasoning_budget_tokens: 4096` | 思考硬上限，到点自动收尾再作答 | 模型模板默认 xhigh，Claude Code 经 new-api 会被映射成 high→xhigh，曾一次想 1.7 万 token、4 分半 |
| `reasoning_close_retry: true` | 想完没正文时自动补一次作答 | 防止 agent 卡住 |
| `reasoning_effort: medium` | 请求未指定档位时的默认值 | 兜底 |
| `fit_max_tokens: true` | prompt + max_tokens 超出上下文时自动缩小 max_tokens | Claude Code 每次带 32000，否则 prompt 超过约 99K 就 400 |
| `host` | 只监听 tailnet IP | 不暴露到局域网/公网 |

保持默认、已实测无提升的：`--spec 4`、`--spec-min-p 0.5`、`--expert-cache auto`、`--kv int8`、`STRATA_EXCHANGE_ROTATE`、`STRATA_ADAPT_OVERLAP`、`--aux-cpus`、`--pool-workers`、CPU performance 调频。`--kv q4_0` / `k8v4` 长文档精度下降，不用。

---

## 7. Tailscale + API key + systemd

### 7.1 Tailscale

```bash
curl -fsSL https://pkgs.tailscale.com/stable/ubuntu/noble.noarmor.gpg | sudo tee /usr/share/keyrings/tailscale-archive-keyring.gpg >/dev/null
curl -fsSL https://pkgs.tailscale.com/stable/ubuntu/noble.tailscale-keyring.list | sudo tee /etc/apt/sources.list.d/tailscale.list >/dev/null
sudo apt-get update && sudo apt-get install -y tailscale
sudo tailscale up --hostname=gpu-linux --accept-dns=false --accept-routes=false   # 打开输出的链接授权
tailscale ip -4                                                                    # 记下 <TAILNET_IP>
```

在 Tailscale 后台给这台机器点 **Disable key expiry**，否则默认 180 天后断线。VPS 也要在同一个 tailnet 里。

### 7.2 API key（不进配置文件、不进命令行参数）

```bash
sudo install -d -m 700 /etc/strata
printf 'STRATA_API_KEY=strata-%s\n' "$(openssl rand -hex 24)" | sudo tee /etc/strata/api.env >/dev/null
sudo chmod 600 /etc/strata/api.env
```

### 7.3 systemd

`/etc/systemd/system/strata.service`：

```ini
[Unit]
Description=Strata LLM server (Qwen3.8-Flash-Next IQ3_XXS)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=<LINUX_USER>
Group=<LINUX_USER>
WorkingDirectory=/home/<LINUX_USER>/Strata
ExecStart=/home/<LINUX_USER>/Strata/run-iq3_xxs.sh
Restart=on-failure
RestartSec=15
TimeoutStartSec=600
TimeoutStopSec=60
LimitMEMLOCK=infinity
LimitNOFILE=65536

[Install]
WantedBy=multi-user.target
```

`/etc/systemd/system/strata.service.d/tailnet.conf`（等 tailnet IP 就绪再启动，并注入 key）：

```ini
[Unit]
After=tailscaled.service
Wants=tailscaled.service

[Service]
EnvironmentFile=/etc/strata/api.env
ExecStartPre=/bin/sh -c 'for i in $(seq 90); do ip -4 addr show tailscale0 2>/dev/null | grep -q "inet <TAILNET_IP>/" && exit 0; sleep 2; done; exit 1'
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now strata
```

`LimitMEMLOCK=infinity` 必须有：启动时要锁定约 33 GiB 专家和 3 GiB KV。首次加载约 35–70 秒，期间机器可能短暂卡顿。

---

## 8. new-api（VPS 上）

在 new-api 管理后台操作（也可以用管理员系统访问令牌调管理 API，写操作发到 master 节点）：

1. **分组**：系统设置里 `GroupRatio` 增加 `"local-gpu": 1`。
   - 只给自己用：`group_special_usable_group` 里在自己所在分组下加 `"+:local-gpu": "本地GPU"`。
   - 开放给所有人：加到 `UserUsableGroups`（不建议，一次只能处理一个请求）。
2. **渠道**：

| 字段 | 值 |
|---|---|
| 类型 | OpenAI |
| Base URL | `http://<TAILNET_IP>:8080` |
| 密钥 | `/etc/strata/api.env` 里的 Strata key |
| 模型 | `qwen3.8-flash-next` |
| 模型映射 | `{"qwen3.8-flash-next": "qwen3.8-flash-next-iq3_xxs"}` |
| 分组 | `local-gpu` |
| 最大并发 | 3（多出来的在 Strata 里排队，不会被拒） |
| 自动禁用 | 关（长思考的慢请求可能被误判） |

3. **定价（必须设）**：`ModelRatio` 里 `"qwen3.8-flash-next": 0.05`，`CompletionRatio` 里 `"qwen3.8-flash-next": 2`。不设的话自用模式会套默认倍率 37.5（约 $75/百万 token）。
4. **令牌**：新建令牌，分组选 `local-gpu`。
5. **连通性**：VPS 宿主机在 tailnet 里时，docker 容器出站会被 MASQUERADE 到 tailnet，能直接访问。验证：

```bash
docker exec new-api wget -t 1 -T 15 -qO- --header="Authorization: Bearer <STRATA_KEY>" http://<TAILNET_IP>:8080/v1/models
```

`STREAMING_FIRST_RESPONSE_TIMEOUT=30` 不会掐断长请求：实测 110K prompt 经 new-api 流式，68 秒后才出首字节，仍正常返回。

---

## 9. 客户端配置

通用参数：

| 项目 | 值 |
|---|---|
| Base URL | `https://<NEWAPI_DOMAIN>/v1`（Anthropic 协议客户端填 `https://<NEWAPI_DOMAIN>`） |
| 模型名 | `qwen3.8-flash-next` |
| 上下文 | 131072 |
| 最大输出 | 16384 |
| 输入 | 仅文本（未加载视觉模块） |
| 能力 | 工具调用、流式、推理（`reasoning_content`）；无视觉、无联网 |
| 推理等级 | none / low / medium / high，默认 medium；high 会被模板升成 xhigh；全部受 4096 上限约束 |
| 超时 | ≥ 300 秒（长 prompt 首字慢） |

**Claude Code**：`ANTHROPIC_BASE_URL=https://<NEWAPI_DOMAIN>`，令牌用 local-gpu 分组的令牌，`claude --model qwen3.8-flash-next`。思考由服务端 4096 上限兜住，客户端不用改。

**ZCode**（`~/.zcode/v2/provider_config.json`，先完全退出 ZCode 再改）：在 `providerRules` 加一个自定义 provider，`api.type` 用 `openai-chat-completions`；在 `manualProviderModelRules` 加：

```json
{
  "modelId": "qwen3.8-flash-next",
  "providerId": "<该 provider 的 UUID>",
  "config": {
    "enabled": true,
    "properties": {
      "contextWindow": 131072,
      "supportsJsonSchemaOutput": false,
      "supportsNativeWebSearch": false,
      "supportsMidConversationSystem": false,
      "inputFormat": {"supportsImage": false, "supportsVideo": false, "supportsPdf": false}
    },
    "optionSpecs": {
      "reasoningLevel": {"values": ["none", "low", "medium", "high"], "map": "{ \"reasoning_effort\": reasoningLevel }"},
      "maxOutputTokens": {"max": 16384}
    }
  }
}
```

**pi**（`~/.pi/agent/`）：`models.json` 加 provider：

```json
"local-gpu": {
  "baseUrl": "https://<NEWAPI_DOMAIN>/v1",
  "api": "openai-completions",
  "models": [{
    "id": "qwen3.8-flash-next",
    "reasoning": true,
    "input": ["text"],
    "contextWindow": 131072,
    "maxTokens": 16384,
    "thinkingLevelMap": {"off": null, "minimal": "low", "low": "low", "medium": "medium", "high": "high", "xhigh": "high", "max": "high"},
    "type": "chat"
  }]
}
```

`auth.json` 加 `"local-gpu": {"type": "api_key", "key": "<令牌>"}`，`settings.json` 的 `enabledModels` 加 `"local-gpu/*"`。使用：`pi --model local-gpu/qwen3.8-flash-next --thinking medium`。

---

## 10. 验收

```bash
# GPU 主机
systemctl is-active strata tailscaled
K=$(sudo sed -n 's/^STRATA_API_KEY=//p' /etc/strata/api.env)
curl -s -o /dev/null -w '%{http_code}\n' http://<TAILNET_IP>:8080/v1/models                              # 401
curl -s -o /dev/null -w '%{http_code}\n' -H "Authorization: Bearer $K" http://<TAILNET_IP>:8080/v1/models   # 200
journalctl -u strata -b | grep -E 'thinking budget|thinking level|ready:'

# 测速（自带工具）
cd ~/Strata && STRATA_KEY=$K .venv/bin/python -I tools/strata_bench.py --url http://<TAILNET_IP>:8080 --runs 3

# 客户端侧
curl -s https://<NEWAPI_DOMAIN>/v1/chat/completions -H "Authorization: Bearer <令牌>" -H 'Content-Type: application/json' -d '{"model":"qwen3.8-flash-next","messages":[{"role":"user","content":"你好"}],"max_tokens":1024}'
```

整机重启后确认 Tailscale 和 Strata 自动起来、new-api 能调通（本次已验证）。

---

## 11. 实测性能（RTX 4080 SUPER + 46 GB DDR4-3200 + ReBAR，IQ3_XXS，128K）

| 测试 | 结果 |
|---|---|
| 短对话 decode | 58.8 tok/s（XMP 前 47.6） |
| agent 多轮 decode | 72.2 tok/s |
| 4K prompt prefill | 1168 tok/s，首字 3.5 s |
| 16K prompt prefill | 1717 tok/s，首字 9.4 s |
| 89K 大海捞针 | 答对，prefill 1540 tok/s，首字约 58 s |
| 111K prompt + max_tokens 32000 | 正常返回（fit_max_tokens） |
| 显存 / 内存 | 15.5 / 16 GB；专家锁页约 32.6 GiB + KV 3 GiB，系统剩约 5 GiB |

---

## 12. 踩坑清单

1. **CUDA 用 13.0，不要 13.2 早期版本**，否则引擎可能输出乱码。
2. **加 NVIDIA 源后必须 pin 驱动包**，否则 `apt upgrade` 会把 Ubuntu 驱动换成 NVIDIA 源的。
3. 系统是中文 locale 时，apt 输出是中文，脚本里 grep 英文关键字前要 `LC_ALL=C`。
4. **`max_tokens` 计入上下文**：Strata 不截断，prompt + max_tokens 超限直接 400 → 开 `fit_max_tokens`。
5. **模型默认 xhigh 思考**，Claude Code 经 new-api 仍是最高档 → 用服务端 `reasoning_budget_tokens` 硬上限（配置里的 `reasoning_effort` 只管没指定档位的请求，管不住 Claude Code）。
6. **new-api 未定价模型按默认倍率 37.5 计费**，一定要手动设 ModelRatio / CompletionRatio。
7. 先确认对外域名的 DNS 真的指向自己的 VPS，nginx 里有 `server_name` 不代表 DNS 解析到这里。
8. Strata 一次只处理一个请求；new-api 渠道并发设 1 时，Claude Code 的并发请求会被直接 503，设 3 让 Strata 排队。
9. 改配置重启时，要等 8080 端口真正释放再起新进程；脚本里 `pkill -f`/`pgrep -f` 会匹配到自己的命令行，用 `[s]erve/server.py` 这种写法。
10. low-RAM 模式下长 prompt（50K+）会有部分专家从 SSD 读，prefill 比内存充足时慢，属预期。
11. memtester 1 轮只是快速检查，混插内存跑 XMP 建议 MemTest86 过夜；出现乱码或异常退出先怀疑 XMP。
12. new-api 是主从节点时，写操作发 master，slave 同步可能有延迟。

---

## 13. 运维

```bash
sudo systemctl status strata       # 状态
sudo systemctl restart strata      # 改配置后重启（约 35 秒）
sudo systemctl stop strata         # 停止，释放约 40 GB 内存和显存
journalctl -u strata -f            # 服务输出
tail -f ~/Strata/strata-iq3_xxs.log   # 引擎日志
```

- 每次改配置前先备份：`cp strata-iq3_xxs.json strata-iq3_xxs.json.bak-$(date +%Y%m%d-%H%M%S)`。
- 升级：`cd ~/Strata && git fetch --tags && git checkout <新tag> && ./setup.sh`；先看 release 说明里有没有改默认值（例如 tail-skip 就是 0.1.42 新加的默认有损项）。
- 换量化：先删旧模型目录和 `packs/` 再装，磁盘放不下两个版本。

## 14. 可选的进一步优化

| 项目 | 效果 | 代价 |
|---|---|---|
| 内存加到 64 GB+ | 更多专家常驻、长 prompt 不读 SSD、可开 256K | 换内存条 |
| `STRATA_ROUTE_TAIL_SKIP=7` | decode 约 +28% | **有损**，输出会变 |
| 思考上限 2048 | 响应更快 | 复杂推理变弱 |
| 换 IQ2_XS / Q2_0 | decode 约 +25–50% | 均分降约 3.5 |
| GPU 主机无桌面运行 | 多出约 230 MiB 显存、1.5 GB 内存 | 不能当桌面用 |
