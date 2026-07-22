# Fin-EKG 工程坑记录（ENGINEERING NOTES）

> 记录**已踩过的坑 + 定论**，用于减少重复问题。新踩坑随手补一条。服务器完整运维见
> [`GPU_RUNBOOK.md`](GPU_RUNBOOK.md)，数据合规见 [`DATASETS.md`](DATASETS.md)。

## GPU / 服务器运维

- **card 3 故障**，需 NVML shim：`CUDA_DEVICE_ORDER=PCI_BUS_ID LD_LIBRARY_PATH=/data/TJK/Fin-EKG/nvmlshim:$LD_LIBRARY_PATH`。card 0/2 常被用户 `Zhyw` 抢，**优先 card 1**。
- **tmux/screen 不在非交互 ssh 的 PATH**。起任务用 `bash -lc` + 绝对 uv 路径 `/home/TJK/.local/bin/uv` + `nohup` 或 `screen -dmS`。screen v4.09 可用。
- **`uv run` 约 1 分钟才真正占显存**。发射后轮询 VRAM 爬升（>1GiB）再判「真在训」，别立刻判死。
- **起长训练前原子 `nvidia-smi` 核卡，且检查结果不能隔着一次启动复用**（检查后卡可能被别人占，探针必 OOM）。**未经明确要求不起长训练。**
- **ssh 间歇性掉线**：`kex_exchange_identification: Connection reset`（gateway 限速，重试可过）/ cpolar 隧道 `Connection refused`（服务器侧隧道后端没起，客户端无解）。→ **三态判活**（ALIVE / GONE / ssh 失败），**只有成功 ssh 读到进程 GONE 才算训练结束**；ssh/工具失败**绝不**当作被观察对象（训练）的结论。
- **`pgrep -af <pat>` 会匹配探针自身命令行**（命令里含该字符串）。用 `[e]valuate_cgep` 括号技巧或核对 PID。
- **服务器不是 git 仓库**（本地是）。同步用 `scp`/`rsync` **指定文件** + `sha256sum` 双端核。**别跑 `rsync --delete`**（会删 remote-only 的 `runs/`、`nvmlshim/`、`scripts/nvml_hide_faulted_gpu.so`）。远端产物在 `/data/TJK/Fin-EKG/runs`。

## 数据

- **MAVEN 版 SeDGPL 数据未发布**（只发 `ESCSubWoRe.npy`）→ 论文 CGEP-MAVEN **27.9 不可比**；主表以自跑 SeDGPL 为准。
- **ESC 必须 topic 交叉验证**；文档级切分泄漏同 topic 故事（SeDGPL 公开 19.6 就是泄漏值）。
- **ICEWS14 已验证=正版 TiRGN，勿重切**；ICEWS 一律用 timestamps **计数切分**。
- **CGEP 词表须 transductive**（覆盖 train+test 的 `<a_i>` token；否则测试全编码失败）。只 token 清单跨切分，无标签/图/梯度泄漏。
- **MAVEN 触发词粘标点（`died.`）/ 大小写不一（`revolution`）** → `token_span` 加「标点+大小写」两级兜底；ESC 有不连续 mention（`keep a hold on`）→ 复刻 `doCorrect` 加宽为连续 span。
- **外部 pickle `ESCSubWoRe.npy` 必须用 `succession.data.esc.load_npy_object` 白名单加载**（安全）。

## 代码 / 评测

- **平坦分数假象**：词表只在 train 建 → 测试全编码失败 → 返回平坦分数 → 乐观 tie-break 下 gold 全排 0 → 假 MRR 1.0。用 `mrr_strict` 戳穿；`UnscorableInstance` 计最差排名 + 单独报 `n_unscorable`，**绝不丢出分母**。
- **查询边判据 = 尾节点出度 0 且入度 1**（不只出度 0）。gold 若出现在其他边会把答案印进 prompt（ESC 1192/1192 成立）。
- **DsGL 截断 = 按存储顺序取前 20 条边**（`EDGE_BUDGET=20`），最短路距离只用于**排序**幸存边（远边在前）。
- **`heuristic._temporal` 默认 `corpus` scope 是真 bug**（99.93% 跨主体伪影）→ 默认改 `subject`。
- **SARGE 无置信度信号** → confidence 不伪造代理分，如实透传 + 告警（曾硬编码 1.0 = bug，已修）。
- **`core.io.event_nodes_from_sarge`**：读 `evidence`（脚本曾写 `argument_evidence`）且保 `metadata`，旧键向后兼容。
- **conformal 改 `fit(train-only)` 致 fixture 指标下降是预期**，不是回归。
- **blackboard 不可变**：agent 只读、在 **copy** 上标注。
- **给已调好的门控编码器加 embedding 输入流，必须 no-op 起步**（M2 结构感知编码，2026-07-17）：默认 `nn.Embedding` 是 N(0,1)（行范数 ~28）→ 碾压融合 `h2`（~8）、init 时把事件 token 的 input embedding 扰动 **185%**（`||h3−h2||/||h2||`=1.85），lr=1e-6 十轮救不回 → **MRR 腰斩 0.1867→0.088**（是 bug、不是「结构有害」的结论）。修＝**zero-init `nn.Embedding` + 门控残差** `h3=h2+g·struct`（`GatedFusion.residual`，y=0 时恒等）→ init 扰动 →0、ON 臂起点＝baseline。诊断 `diag_m2.py` 量 `||h3−h2||/||h2||`。对照 SeDGPL 对 `<a_i>` 新行专门 mean-init 同理。

## 纪律（硬约束）

- 本地 `uv run pytest` = **330 passed / 14 skipped**（M2 后；torch 门控测试本地 skip、服务器跑），**只增不改**；`uv run ruff check src tests scripts` **0 error（≤100 列）**。
- 包/函数名**不得含 `ch1/ch2/ch3`**；新组件走 registry + lazy import；GPU 组件配 CPU 缓存回放。
- **`EventNode` schema 零新增字段**（扩展用 `metadata`）；`CgepNode` 可加字段。
- 不可改的测试锁：`tests/core/test_propagation.py`。
- 报告结果**如实**：数字降就说降；专利 / 论文写作不在计划范围。
