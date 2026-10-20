# 工具

都假设 GPU 主机是 Linux，Strata 装在主机用户的 `~/Strata`，systemd 服务名为 `strata`（见 [docs/full-chain.md](../docs/full-chain.md)）。脚本不读 API key 和对话内容。

| 脚本 | 在哪运行 | 作用 | 要停服吗 |
|---|---|---|---|
| `strata_cache_stats.sh` | 客户端（经 ssh） | 统计某一次启动以来的 ≥32K 冷读次数和耗时、会话缓存停放 / 恢复 / 被拒次数、引擎堆内存 RSS 和缓存字节数 | 不用 |
| `strata_ab.py` | GPU 主机 | 两个或多个 Strata 版本 / 开关组合轮流启动，对比 decode、prefill 速度和输出 | 要 |
| `strata_cache_soak.py` | GPU 主机 | 多个长会话轮流追加，按线上缓存参数对比各版本的恢复命中、驱逐和堆内存 | 要 |

## strata_cache_stats.sh

```bash
STRATA_HOST=<ssh 名称> tools/strata_cache_stats.sh      # 当前这次启动（STRATA_HOST 默认 gpu-linux）
STRATA_HOST=<ssh 名称> tools/strata_cache_stats.sh 2    # 上一次启动
```

引擎堆 RSS 远大于缓存字节数、停放被 `physical RAM` 拒绝很多，说明 glibc 没把释放的快照内存还回去（见 full-chain §6 的 `MALLOC_MMAP_THRESHOLD_`）。

## strata_ab.py / strata_cache_soak.py

两个脚本放在同一目录（soak 会 import ab）。它们在服务运行时拒绝启动，测试服务只监听 `127.0.0.1:8090`，所有组都从同一份冻结的 expert profile 启动，不写回线上安装。

```bash
sudo systemctl stop strata

# 速度和输出对比（每组：NAME:root=<Strata 目录>,exe=<引擎>,[xargs=参数;参数],[环境变量=值...]）
python3 -I tools/strata_ab.py --workdir ~/strata-ab --rounds 5 \
  A:root=$HOME/Strata,exe=$HOME/Strata/engine/strata \
  B:root=$HOME/Strata-new,exe=$HOME/Strata-new/build/strata

# 输出逐字节对比：加 --identity --slots N（N 取各组 auto 槽数里较小的那个）

# 会话缓存浸泡
python3 -I tools/strata_cache_soak.py --workdir ~/strata-soak --rounds 6 \
  A:root=$HOME/Strata,exe=$HOME/Strata/engine/strata \
  B:root=$HOME/Strata-new,exe=$HOME/Strata-new/build/strata

sudo systemctl start strata
```

注意：Strata 在 greedy 下也不是逐次确定的（多 token CPU 内核的舍入随 MTP drafting 变化，Strata DETAILS.md「Reproducible greedy output」），默认参数下同一版本几轮之间输出就可能不同。`strata_ab.py` 的报告因此先列每组自己的轮间稳定性，再和基准组比。
