# EKG 工程坑记录（ENGINEERING NOTES）

> 记录**已踩过的坑 + 定论**，用于减少重复问题。新踩坑随手补一条。服务器完整运维见
> [`GPU_RUNBOOK.md`](GPU_RUNBOOK.md)，数据合规见 [`DATASETS.md`](DATASETS.md)。

## GPU / 服务器运维

- **card 3 故障**，需 NVML shim：`CUDA_DEVICE_ORDER=PCI_BUS_ID LD_LIBRARY_PATH=/data/TJK/ekg/nvmlshim:$LD_LIBRARY_PATH`。card 0/2 常被用户 `Zhyw` 抢，**优先 card 1**。
- **tmux/screen 不在非交互 ssh 的 PATH**。起任务用 `bash -lc` + 绝对 uv 路径 `/home/TJK/.local/bin/uv` + `nohup` 或 `screen -dmS`。screen v4.09 可用。
- **`uv run` 约 1 分钟才真正占显存**。发射后轮询 VRAM 爬升（>1GiB）再判「真在训」，别立刻判死。
- **起长训练前原子 `nvidia-smi` 核卡，且检查结果不能隔着一次启动复用**（检查后卡可能被别人占，探针必 OOM）。**GPU 使用无需逐次点头**（作者授权），但不挤占他人正在跑的卡。
- **ssh 间歇性掉线**：`kex_exchange_identification: Connection reset`（gateway 限速，重试可过）/ cpolar 隧道 `Connection refused`（服务器侧隧道后端没起，客户端无解）。→ **三态判活**（ALIVE / GONE / ssh 失败），**只有成功 ssh 读到进程 GONE 才算训练结束**；ssh/工具失败**绝不**当作被观察对象（训练）的结论。
- **`pgrep -af <pat>` 会匹配探针自身命令行**（命令里含该字符串）。用 `[e]valuate_cgep` 括号技巧或核对 PID。
- **服务器不是 git 仓库**（本地是）。同步用 `scp`/`rsync` **指定文件** + `sha256sum` 双端核。**别跑 `rsync --delete`**（会删 remote-only 的 `runs/`、`nvmlshim/`、`scripts/nvml_hide_faulted_gpu.so`）。远端产物在 `/data/TJK/ekg/runs`。

## 数据

- **v4 主数据状态以 `DATASET_SURVEY.md` + `data/raw/DATA_PROVENANCE.md` 为准**。MAVEN-FACT
  train/valid 已就位；MATRES/RAMS/WikiEvents/ECB+ 目前只有 raw，未生成项目 processed 输出。
- **MAVEN 版 SeDGPL 数据未发布**（只发 `ESCSubWoRe.npy`）→ 论文 CGEP-MAVEN **27.9 不可比**；主表以自跑 SeDGPL 为准。
- **ESC 必须 topic 交叉验证**；文档级切分泄漏同 topic 故事（SeDGPL 公开 19.6 就是泄漏值）。
- **ICEWS/FinDKG 只属冻结 TKG 线**；从 tag 复现时保持 release split，ICEWS 用 timestamps
  **计数切分**，结果不得混入 v4 主表。
- **CGEP 词表须 transductive**（覆盖 train+test 的 `<a_i>` token；否则测试全编码失败）。只 token 清单跨切分，无标签/图/梯度泄漏。
- **MAVEN 触发词粘标点（`died.`）/ 大小写不一（`revolution`）** → `token_span` 加「标点+大小写」两级兜底；ESC 有不连续 mention（`keep a hold on`）→ 复刻 `doCorrect` 加宽为连续 span。
- **外部 pickle `ESCSubWoRe.npy` 必须用 `succession.data.esc.load_npy_object` 白名单加载**（安全）。

## 代码 / 评测

- **平坦分数假象**：词表只在 train 建 → 测试全编码失败 → 返回平坦分数 → 乐观 tie-break 下 gold 全排 0 → 假 MRR 1.0。用 `mrr_strict` 戳穿；`UnscorableInstance` 计最差排名 + 单独报 `n_unscorable`，**绝不丢出分母**。
- **查询边判据 = 尾节点出度 0 且入度 1**（不只出度 0）。gold 若出现在其他边会把答案印进 prompt（ESC 1192/1192 成立）。
- **DsGL 截断 = 按存储顺序取前 20 条边**（`EDGE_BUDGET=20`），最短路距离只用于**排序**幸存边（远边在前）。
- **`heuristic._temporal` 默认 `corpus` scope 是真 bug**（99.93% 跨主体伪影）→ 默认改 `subject`。
- **conformal 改 `fit(train-only)` 致 fixture 指标下降是预期**，不是回归。
- **blackboard 不可变**：agent 只读、在 **copy** 上标注。
- **给已调好的门控编码器加 embedding 输入流，必须 no-op 起步**（M2 结构感知编码，2026-07-17）：默认 `nn.Embedding` 是 N(0,1)（行范数 ~28）→ 碾压融合 `h2`（~8）、init 时把事件 token 的 input embedding 扰动 **185%**（`||h3−h2||/||h2||`=1.85），lr=1e-6 十轮救不回 → **MRR 腰斩 0.1867→0.088**（是 bug、不是「结构有害」的结论）。修＝**zero-init `nn.Embedding` + 门控残差** `h3=h2+g·struct`（`GatedFusion.residual`，y=0 时恒等）→ init 扰动 →0、ON 臂起点＝baseline。诊断 `diag_m2.py` 量 `||h3−h2||/||h2||`。对照 SeDGPL 对 `<a_i>` 新行专门 mean-init 同理。

## 环境 / 工具链

- ⛔ **服务器上跑 `uv run` / `uv sync` 会拆掉已验证的 GPU 栈**（2026-07-27 `--dry-run` 实测）：远端 venv
  装的是 `llm+serve+rl+gnn+dev` 全套 extras，而 uv 会把环境**对齐到你给出的 extras 集合、卸掉多余的**。
  裸 `uv sync` → **卸 165 个包**（torch/transformers/vllm/trl/peft/bitsandbytes 全没）；
  `--extra llm` → **仍卸 109 个**（vllm/trl/ray/xformers/torchvision/torch-geometric）；全套五个 extra
  才只卸 `patchelf`/`setuptools` 构建残留。恢复要重下数 GB。**服务器一律 `.venv/bin/python`**；
  非要用 uv 就 `uv run --no-sync`。`uv pip install -e . --no-deps` 是安全的。
  ⚠️ 这条曾潜伏在 `PHASE_B_HANDOFF` Step 1 与服务器待机脚本里（都是裸 `uv run python`）——
  只因待机脚本 `cd` 到改名前的旧路径先 exit 2，环境才没被拆。
- **测试计数两端不同，不是回归**：本地无 torch = **241 passed / 12 skipped**；服务器有 torch =
  **252 passed / 1 skipped**（`test_model_skip.py` 的 skip 条件是反的：本地 skip「需要 torch」，
  服务器 skip「torch 已安装」）。判回归要跟同一端的基线比。
- **项目目录改名后必须 `uv sync --extra dev --reinstall`，只跑 `uv sync` 不够**（2026-07-27，
  `Fin-EKG` → `ekg`）：`.venv/bin/` 里第三方 console script 的 shebang 是**安装时写死的绝对路径**
  （`#!/…/Fin-EKG/.venv/bin/python3`），`uv sync` 不会重写已存在的脚本 → `uv run pytest` 报
  **`ModuleNotFoundError: No module named 'pydantic'`**（31 collection errors），看着像依赖坏了。
  **机制**：shebang 解释器不存在 → `execve` 返回 ENOENT → `execvp` 当成「这个文件没找到」**继续沿
  PATH 找下一个同名程序**，于是跑到了 `~/.local/bin/pytest`（`#!/usr/bin/python3`，系统 python，
  没有本项目依赖）。所以症状不是「bad interpreter」而是「装好的包全都 import 不到」。
  **判据**：`uv run python -m pytest` 正常而 `uv run pytest` 挂 ⇒ 一定是 shebang，不是环境。
  实测受影响 9 个：`pytest` `py.test` `coverage{,3,-3.10}` `dotenv` `f2py` `pygmentize` `tqdm`；
  本项目自己的 `ekg-smoke` 因改名时重装而正常。**服务器 pull 后同理。**
- **服务器上改名不能靠 `--reinstall`**（会触发 §0 的拆栈），要**手工重定位 venv**（2026-07-27 实操，
  `/data/TJK/Fin-EKG` → `/data/TJK/ekg`，56 个 shebang + editable 安装全废，`import ekg` 与
  `import finekg` 双双失败）：
  ```bash
  cd /data/TJK/ekg
  # 1) shebang：只改第 1 行，跳符号链接与无 shebang 的二进制（ruff/ninja/patchelf 别 sed！）
  for f in .venv/bin/*; do [ -L "$f" ] && continue; [ -f "$f" ] || continue
    head -1 "$f" | grep -q '^#!/data/TJK/Fin-EKG/' || continue
    sed -i '1s|^#!/data/TJK/Fin-EKG/|#!/data/TJK/ekg/|' "$f"; done
  # 2) activate 系列的 VIRTUAL_ENV 路径 + pyvenv.cfg 的 prompt
  # 3) editable 安装（包名也变了）：旧 dist-info 必须先卸
  export VIRTUAL_ENV=/data/TJK/ekg/.venv
  /home/TJK/.local/bin/uv pip uninstall finekg
  /home/TJK/.local/bin/uv pip install -e . --no-deps     # --no-deps = 不碰 torch/vllm
  ```
  editable 的 `_editable_impl_<name>.pth` 里就是一行绝对路径 `<root>/src`；dist-info 名跟着包名变
  （`finekg-0.1.0.dist-info` → `ekg-0.1.0.dist-info`），旧入口脚本 `finekg-smoke` 也要一并消失。
  验收：`.venv/bin/python -c "import ekg, torch"` + `.venv/bin/pytest` + `.venv/bin/ekg-smoke`。
- **改名后 `__pycache__` 里的 bytecode 仍嵌旧路径**（`co_filename`）→ traceback 显示已不存在的目录、
  源码行渲染成 `???`，误导定位。改名/搬目录后先
  `find . -name __pycache__ -not -path "./.venv/*" -exec rm -rf {} +` 再判断。
- **remote-only 残留会悄悄留在 docs/**：服务器 `docs/{chapter1,midterm}`（32M，历史 scp 遗留、untracked）
  `git reset --hard` 不会删。改文档结构后要在服务器侧核一遍 `ls docs/`，多出来的移到
  `/data/TJK/ekg-backup-20260727/`（**不要 `git clean -fdx`**，会连 `runs/`/`data/` 一起删）。

## 纪律（硬约束）

- 主干验证（2026-07-27 重构后）：本地 `uv run pytest` = **241 passed / 12 skipped**（skip 均为本地
  无 torch 的神经门控测试）；`uv run ruff check src tests scripts` **0 error（≤100 列）**；
  `ekg-smoke` 通过。测试计数随旧线移出主干而变化，不得拿旧计数（239/11、269/12）判断当前回归。
- 包/函数名**不得含 `ch1/ch2/ch3`**；新组件走 registry + lazy import；GPU 组件配 CPU 缓存回放。
- **`EventNode` schema 零新增字段**（扩展用 `metadata`）；`CgepNode` 可加字段。
- 不可改的测试锁：`tests/core/test_propagation.py`。
- 报告结果**如实**：数字降就说降；专利 / 论文写作不在计划范围。
