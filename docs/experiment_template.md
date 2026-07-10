# MaxWell 闭环 / 在线实验 — 代码生成 Prompt（通用模板）

> **用途**：每次新建或改写实验代码时，将本文全文（或对应章节）粘贴给 AI，作为必须遵循的规范。  
> **参考 demo**：`0520test_closed_stim/`（Python 配置 + C++ 实时检测 + 闭环发刺激 + H5 录制）。

---

## 一、实验目标描述（每次实验前先填写）

请在生成代码前，先明确并写入 `README.md`：

1. **实验名称**（英文 snake_case，用于目录与 H5 文件名前缀）
2. **细胞信息**：细胞编号、实验日期、培养天数（DIV）
3. **科学问题 / 假设**：本次要验证什么（例：网络累计 spike 达阈值后单脉冲反馈能否改变后续 burst 节律）
4. **闭环逻辑**（若适用）：
   - 检测信号来源（全阵列 spike 计数 / 指定电极 / burst 检测 / 自定义算法）
   - 触发条件与阈值（例：每累计 N spikes 发 1 次刺激）
   - 停止条件（例：刺激满 M 次 / 总时长 / 手动 Ctrl+C）
5. **记录结构**：一次 run 含几组 **block**（按实验条件命名，如不同电极组/幅值/协议）；**每组 block 固定三段** `01` 自发 → `02` 刺激 → `03` 自发
6. **预期输出**：H5 命名、是否需要 events 标注、后续分析脚本

---

## 二、项目目录结构（必须遵守）

每次实验是一个独立 Python 包，目录如下：

```text
实验名称/
├── README.md                 # 实验计划、参数说明、运行方式、已知坑（由 prompt + config 自动生成/更新）
├── requirements.txt          # Python 依赖；启动时检查，缺失则 log 报错并给出修复建议
├── setup.py                  # 将项目安装为可 import 的包（python/ 下的模块可被 main 引用）
├── .gitignore                # 忽略 data/、build/、__pycache__、*.h5 等
├── main.py                   # ★ 一键运行入口（唯一推荐的用户启动方式）
├── config/
│   ├── system.yaml           # 系统级配置（见第三节）
│   └── stimulation.yaml      # 刺激协议与电极组（见第四节）
├── cpp/
│   ├── CMakeLists.txt
│   ├── src/                  # C++ 实时检测 / 闭环触发源码
│   └── build/                # cmake 产物（可为空；main 可自动 cmake --build）
├── python/
│   ├── __init__.py
│   ├── experiment_runner.py  # 实验主流程（从 main 调用）
│   ├── maxwell_setup.py      # 阵列、录制、Sequence 构建
│   └── utils/                # burst 检测、波形提取、关键电极筛选等（可复用算法）
├── scripts/
│   └── build_cpp.sh          # 可选：手动编译 C++
└── data/                     # 默认数据根（system.yaml 可覆盖）
    └── {YYYYMMDD_HHMMSS}_data/   # ★ 每次 main 执行新建一个 run 目录
        ├── {HHhMMmSSs}.cfg     # 从 system.yaml 指定路径复制过来的电极 map
        ├── log.txt             # 本次运行完整 log（stdout/stderr 重定向或 logging 写入）
        ├── external_time_table.csv   # 可选：run 级审计总表（见第九节，非分析主索引）
        ├── external_time_table.json  # 同上，JSON 镜像（可选）
        ├── config_snapshot/    # 可选：复制本次使用的 system.yaml + stimulation.yaml
        └── raw_data/
            ├── {block_A}/              # 第一组实验条件（命名见 config，如 group_A_150mV）
            │   ├── 01_pre_spont/       # 刺激前自发
            │   ├── 02_stim/            # 刺激 / 闭环段
            │   │   ├── *.raw.h5
            │   │   └── stim_times.txt  # ★ 本段刺激时间表（相对本段 H5 起点，秒）
            │   └── 03_post_spont/      # 刺激后自发
            ├── {block_B}/
            │   ├── 01_pre_spont/
            │   ├── 02_stim/
            │   └── 03_post_spont/
            └── ...                     # 多组 block 并列；每组内固定 01/02/03 三段
```

**硬性规则**

- **一次 `main.py` 执行 = 一次实验 run**；禁止覆盖已有 `{时间}_data/` 目录。
- run 目录名格式：`YYYYMMDD_HHMMSS_data`（例：`20260525_014300_data`），取 **main 开始时刻**（本地时区）。
- 保存路径 = `system.yaml` 的 `data_root`（缺省 `./data`）+ 上述 run 子目录。
- 所有生成物（H5、log、cfg 副本、config 快照、**外界时间总表**、附属 txt/csv）必须落在该 run 目录内，便于追溯。
- **每个含刺激的 phase（通常 `02_stim`）必须生成一份** `stim_times.txt`（见第九节），与对应 `.raw.h5` 同目录；时间为**相对该 H5 段起点**的秒，可直接用于 Raster / batch 分析。
- run 级 `external_time_table.csv` 仅作**审计与追溯**（可选），**不得**作为与单段 H5 对齐的唯一来源。

---

## 三、`config/system.yaml` 字段规范

```yaml
# 示例 — 生成代码时按实际实验填写
culture:
  id: "28937"              # 细胞编号
  div: 35                  # 培养天数
experiment:
  date: "2026-05-25"       # 实验当天
  name: "closed_loop_spike_threshold"   # 当前实验名称
  recording_name_prefix: "cl_spike10k" # H5 文件名前缀

electrode_map:
  cfg_path: "/home/maxwell/configs/260520/16h08m21s.cfg"  # 源 cfg；运行时复制到 run 目录

data:
  root: "./data"           # 数据根；若无则默认 ./data

maxwell:
  device: "maxone"         # maxone | maxtwo
  event_threshold: 8.5
  amplifier_gain: 512

burst_detection:           # 若闭环/分析用到 burst 检测，在此声明默认参数
  bin_ms: 10
  smooth_sigma_ms: 300
  k_rms: 1.2

closed_loop:               # 若适用；纯开环实验可省略或留空
  cpp_runner: "spike_threshold_runner"
  spike_step: 10000        # 累计 spike 阈值
  max_stims: 10
  sequence_name: "spike_10k_closed_loop"
```

---

## 四、`config/stimulation.yaml` 字段规范

### 4.1 电极组

刺激电极按 **组** 管理；不分组则整组视为一组。

```yaml
electrode_groups:
  - name: "group_A"
    electrodes: [7317]
  - name: "group_B"
    electrodes: [7317, 8820]   # 多电极时按 MaxWell 路由能力连接
```

### 4.2 刺激协议三类

命名规则：**名称 + 协议类别 + 参数**。使用时指定「电极组 + 协议名称」。

| 类别 | 标识 | 说明 |
|------|------|------|
| ① single pulse | `single_pulse` | 单个双相脉冲 |
| ② individual burst | `individual_burst` | 单个 burst（burst 内多脉冲） |
| ③ sequence with burst | `sequence_with_burst` | 多个 burst 组成的序列 |

### 4.3 参数（按类别）

**所有协议共有**

| 参数 | 字段名 | 说明 |
|------|--------|------|
| 刺激幅值 | `amplitude_mv` | mV；通过 `query_DAC_lsb_mV()` 换算 DAC 半幅 bit |
| 脉宽 | `pulse_width_us` | 单相脉宽 (µs) |
| 相间间隔 | `inter_phase_interval_us` | 双相脉冲两相之间间隔 (µs)，可选 |

**② individual_burst、③ sequence_with_burst 额外**

| 参数 | 字段名 | 说明 |
|------|--------|------|
| burst 内刺激频率 | `pulse_frequency_hz` | Pulse Frequency |
| burst 内脉冲数量 | `pulses_per_burst` | Number of Pulses per Burst |
| burst 内脉冲间隔 | `interpulse_interval_ms` | 与频率二选一即可（有频率可计算） |

**③ sequence_with_burst 额外**

| 参数 | 字段名 | 说明 |
|------|--------|------|
| burst 数量 | `burst_count` | |
| burst 间隔 | `burst_interval_ms` | 与 burst 频率二选一 |

### 4.4 示例

```yaml
protocols:
  - name: "feedback_single_150mV"
    type: single_pulse
    amplitude_mv: 150
    pulse_width_us: 300
    inter_phase_interval_us: 0

  - name: "train_burst_20hz"
    type: individual_burst
    amplitude_mv: 150
    pulse_width_us: 200
    pulse_frequency_hz: 20
    pulses_per_burst: 10

  - name: "closed_loop_default"
    type: single_pulse
    amplitude_mv: 150
    pulse_width_us: 300
    # 绑定到 closed_loop.sequence_name 使用的 Sequence
```

---

## 五、MaxWell 闭环 Demo 核心逻辑（必须内化）

参考 `0520test_closed_stim/setup_test_closed_loop.py` + `src/spike_threshold_runner.cpp`。

### 5.1 分工

| 层 | 职责 |
|----|------|
| **Python** | 读 yaml → 建 run 目录 → 初始化 MaxLab → 加载 cfg → 路由刺激电极 → 构建 `mx.Sequence`（含 `mx.Event`）→ 开录制 → **启动 C++ 子进程** → 等待结束 → 停录制 |
| **C++** | `DataStreamerFiltered_open` → 逐帧读 spike 计数 → 达阈值 `sendSequence(sequence_name)` → 达 max_stims 退出 |

### 5.2 刺激幅值与 DAC（强制）

```python
lsb = float(mx.query_DAC_lsb_mV())
half_bits = int(round(abs(amplitude_mv) / lsb))   # 1..511
# 双相脉冲：512 - half_bits → delay → 512 + half_bits → delay → 512
```

与 `chip.py` / `StimulationUnit` 文档一致；**禁止**硬编码 magic DAC 值而不查 LSB。

### 5.3 H5 events 写入（强制）

`mx.Event(well_id, event_type, user_id, properties)` 的 **properties 必须是 key-value 成对**（按空格分词后为**偶数个** token）。

```python
# ✗ 错误 — 3 个 token，events 静默不写 H5
mx.Event(0, 1, 10, "spike threshold stim")

# ✓ 正确
mx.Event(0, 1, 10, "type stim")
mx.Event(0, 1, 10, "name feedback_pulse")
```

每个刺激 Sequence 在发脉冲前 `append(mx.Event(...))`；

### 5.4 推荐时序

1. **先** `cmake --build` 确认 C++ 可执行文件存在  
2. `mx.initialize()` → 刺激电源 → gain → threshold  
3. `array.load_config(cfg)` → `connect_electrode_to_stimulation` → `download` → `offset`（含 MX1/MX2 wait）  
4. 构建 persistent `Sequence`（含 Event + DAC 波形）  
5. `mx.Saving()` → `open_directory(run_dir/raw_data/{block}/{phase}/)` → `start_file` → `start_recording`（每个 block 循环 01→02→03）  
6. `subprocess.Popen(cpp_runner, ...)`  
7. `proc.wait()` → `stop_recording` → `stop_file`  
8. C++ 非 0 退出时视为实验失败，log 记录

### 5.5 C++ 编译

- `MAXLAB_ROOT` 指向 `maxlab_lib`（含 `maxlab/include`、`maxlab/lib`）
- Conda 环境优先链接其 `libstdc++.so`（见 demo `CMakeLists.txt`）
- `main.py` 可在实验前自动：`cmake -S cpp -B cpp/build && cmake --build cpp/build`

---

## 六、`main.py` 必须实现的行为

```text
1. 解析命令行（可选 --config-dir、--dry-run）
2. 加载 config/system.yaml + config/stimulation.yaml
3. 检查 requirements.txt（见第七节）→ 结果写入 log
4. 创建 run 目录：{data.root}/{YYYYMMDD_HHMMSS}_data/
5. 复制 cfg → run/{HHhMMmSSs}.cfg；快照 yaml → run/config_snapshot/
6. 配置 logging → run/log.txt（同时 console）
7. 调用 python/experiment_runner.run(...)；每个 `02_stim` 段结束即写入 **`stim_times.txt`**；run 结束可选写入 `external_time_table.csv`
8. 异常时 log traceback，仍尝试写入已收集的各段 `stim_times.txt` 与审计表，exit code != 0
9. 正常结束打印 run 目录绝对路径
```

---

## 七、`requirements.txt` 与环境检查

`main.py` 启动时执行检查逻辑（可放在 `python/utils/env_check.py`）：

1. 读取 `requirements.txt` 每行包名  
2. 尝试 `import` 或 `importlib.metadata.version`  
3. **缺失**：log 明确列出缺失包 + 建议 `pip install -r requirements.txt` 或 conda 环境名  
4. **齐全**：log 一行 `环境检查: OK`  
5. 额外检查：`maxlab` 可 import、`MAXLAB_ROOT` / C++ runner 是否存在（闭环实验）

典型依赖：

```text
maxlab
h5py
numpy
matplotlib
pyyaml
```

---

## 八、数据块目录约定（`raw_data/`）

### 两层结构（不要混淆）

| 层级 | 含义 | 命名 |
|------|------|------|
| **block**（外层） | 一组完整实验条件 | 由多组实验决定，如 `group_A_150mV`、`rank01_el_267`、`freq_10hz` |
| **phase**（内层） | 该 block 内固定三段录制 | 固定为 `01_pre_spont` → `02_stim` → `03_post_spont` |

一次 `main` 运行可在 `raw_data/` 下顺序执行 **多个 block**；每个 block 内 **必须** 按 01 → 02 → 03 录制（自发 → 刺激/闭环 → 自发）。

```text
raw_data/
├── group_A_150mV/                 # block：实验条件 1
│   ├── 01_pre_spont/              # phase：刺激前自发（如 5 min）
│   │   └── group_A_150mV_01_pre_spont.raw.h5
│   ├── 02_stim/                   # phase：刺激或闭环运行段
│   │   └── group_A_150mV_02_stim.raw.h5
│   └── 03_post_spont/             # phase：刺激后自发
│       └── group_A_150mV_03_post_spont.raw.h5
├── group_B_200mV/                 # block：实验条件 2
│   ├── 01_pre_spont/
│   ├── 02_stim/
│   └── 03_post_spont/
└── ...
```

### 命名与 meta

- **block 名**：写在 `config/system.yaml` 的 `experiment.blocks` 列表，或 `stimulation.yaml` 中「电极组 + 协议」组合推导  
- **H5 文件名**：`{recording_name_prefix}_{block}_{phase}.raw.h5`（或与 MaxWell `start_file` 一致）  
- **block 级 meta**：`raw_data/{block}/block_meta.yaml`（电极组、协议名、幅值、时长等）  
- **phase 级 meta**（可选）：`raw_data/{block}/{phase}/phase_meta.yaml`（该段时长、是否闭环、C++ 参数等）
- **刺激时间表**（`02_stim` 强制）：`raw_data/{block}/02_stim/stim_times.txt`（见第九节）

### `system.yaml` 中 block 列表示例

```yaml
experiment:
  blocks:
    - name: "group_A_150mV"
      electrode_group: "group_A"
      protocol: "feedback_single_150mV"
      phases:
        - id: "01_pre_spont"
          duration_s: 300
        - id: "02_stim"
          mode: "closed_loop"    # 或 open_loop / manual
        - id: "03_post_spont"
          duration_s: 300
    - name: "group_B_200mV"
      electrode_group: "group_B"
      protocol: "feedback_single_200mV"
      phases: [...]
```

---

## 九、刺激时间记录（强制）

### 9.1 原则（分析对齐用「段内时间表」）

| 层级 | 文件 | 用途 |
|------|------|------|
| **段级（主）** | `raw_data/{block}/02_stim/stim_times.txt` | 与**该段** `.raw.h5` 一一对应；时间为**相对本段 `record_start`** 的秒，可直接粘贴到 interactive pipeline「补充刺激」或供分析脚本读取 |
| **run 级（辅）** | `external_time_table.csv` | 整场实验审计、墙钟追溯；**禁止**单独用它推算某段 H5 内的刺激时刻 |

**硬性规则**

- **一份 H5（一段录制）↔ 一份 `stim_times.txt`**（仅当该段含刺激；通常 `02_stim`）。
- `stim_times.txt` 中的数字必须与 spike 时间轴一致（相对段起点，约 `0` … `duration_sec`），**禁止**写入整场 run 的 `offset_sec` 或 Unix `epoch_sec`。
- 在 `record_start(02_stim)` 之后、`record_end(02_stim)` 之前发生的每一次 `stim_send`，都必须进入**该段** `stim_times.txt`。
- `01_pre_spont` / `03_post_spont` 若无刺激，**不生成** `stim_times.txt`。

仍须记录墙钟事件（见 9.6），但分析对齐以段内文件为准。

### 9.2 段内时间原点（与 H5 一致）

对每个 phase 目录单独定义原点：

```text
segment_time_origin_epoch = 该段 start_recording() 时刻的 time.time()
```

段内刺激相对秒：

```text
stim_time_sec[i] = stim_send_epoch_sec[i] - segment_time_origin_epoch
```

生成 `stim_times.txt` 时：

1. 仅收集 **同一 `block` + 同一 `phase`（02_stim）+ 同一 `segment_name`** 的 `stim_send` 行。
2. 按 `stim_time_sec` 升序写入。
3. 段结束（`stop_file` 之后）立即写盘，**不要**等到整场 run 结束再统一换算（避免漏记、错位）。

可与 H5 `events.frameno / fs`（相对 `frame_origin`）交叉验证；偏差 > 50 ms 须在 `log.txt` 记 WARNING。

### 9.3 `stim_times.txt` 路径与格式（强制）

**路径（固定）**

```text
raw_data/{block}/02_stim/stim_times.txt
```

与 H5 同目录，例如：

```text
raw_data/block01_burst_outside/02_stim/
├── block01_burst_outside_02_stim.raw.h5
└── stim_times.txt
```

**文件格式**

- 编码 UTF-8；扩展名固定 `.txt`。
- 以 `#` 开头的行为注释（分析/UI 读取时忽略）。
- 数值行为刺激时刻（秒），**相对本段 H5 起点**；一行一个，或同一行逗号/空格分隔（二选一，勿混用两种风格）。
- 注释头至少包含：`block`、`phase`、`segment_name`、`record_start_epoch`、`pulse_count`、`generated_utc`。

**示例**

```text
# stim_times.txt — times relative to THIS segment start (seconds)
# block: block01_burst_outside
# phase: 02_stim
# segment_name: block01_burst_outside_02_stim
# record_start_epoch: 1779733342.8795738
# pulse_count: 20
# generated_utc: 2026-05-26T02:22:54Z
29.149397
29.149522
29.149568
...
29.150174
```

**在 interactive pipeline 中使用**

1. 读取该段 `02_stim/*.raw.h5`；
2. 打开同目录 `stim_times.txt`，复制数值行到「二、补充刺激」；
3. 点击「应用补充刺激」→ Raster 即可看到刺激线。

若 H5 内已有可信 `events`，仍**必须**写 `stim_times.txt`（可从 events 导出），保证「打开文件夹即能分析」。

### 9.4 可选：段级 `segment_time_meta.json`

与 `stim_times.txt` 同目录，机器可读元数据（可选但推荐）：

```json
{
  "block": "block01_burst_outside",
  "phase": "02_stim",
  "segment_name": "block01_burst_outside_02_stim",
  "h5_basename": "block01_burst_outside_02_stim.raw.h5",
  "record_start_epoch": 1779733342.8795738,
  "record_end_epoch": 1779733374.0381918,
  "stim_times_sec": [29.149397, 29.149522],
  "pulse_count": 20
}
```

`stim_times_sec` 须与 `stim_times.txt` 数值一致。

### 9.5 禁止事项（针对旧规范）

| 禁止 | 原因 |
|------|------|
| 仅用 run 级 `external_time_table.csv` 的 `offset_sec` 填手动刺激 | `offset_sec` 相对**整场实验**起点，与单段 H5（0…段长）不在同一坐标系 |
| 把整场 run 所有 block 的 `stim_send` 写入一张总表当作分析输入 | 无法与「当前打开的 H5」一一对应 |
| 刺激写在 `segment_name` 与 H5 不符的行（如 `burst_end`）却不换算到段内秒 | 导致 Raster 对不上 |

### 9.6 run 级审计表（可选，非分析主索引）

路径：

```text
{data.root}/{YYYYMMDD_HHMMSS}_data/external_time_table.csv
```

- 可另存 `external_time_table.json`；内容与 CSV 一致。
- 用于实验审计、墙钟追溯、与 lab notebook 对照；**分析脚本默认读取段内 `stim_times.txt`**。
- 实现：`python/utils/time_log.py` 的 `ExternalTimeLog` 在 run 结束 `save()`；**另**在 `experiment_runner` 每段 `02_stim` 结束调用 `save_segment_stim_times(phase_dir, ...)` 写 `stim_times.txt`。

**CSV 列（审计表，最低要求）**

| 列名 | 说明 |
|------|------|
| `event_type` | `experiment_start` / `record_start` / `record_end` / `stim_send` / `experiment_end` |
| `block` | block 名 |
| `phase` | `01_pre_spont` / `02_stim` / `03_post_spont` |
| `segment_name` | H5 basename（无扩展名） |
| `wall_time_utc` / `wall_time_local` | 墙钟 ISO8601 |
| `epoch_sec` | `time.time()` |
| `offset_sec` | 相对 run 级 `time_origin_epoch`（**仅审计**） |
| `stim_index` / `amplitude_mv` / `electrodes` / `note` | 可选 |

### 9.7 代码模板（Python）

```python
import time
from pathlib import Path
from utils.time_log import ExternalTimeLog, SegmentStimLog

run_origin = time.time()
audit_log = ExternalTimeLog(run_dir=run_dir, time_origin_epoch=run_origin)

# ---------- 02_stim 段开始 ----------
phase_dir = run_dir / "raw_data" / block / "02_stim"
segment_origin = time.time()
segment_log = SegmentStimLog(
    block=block,
    phase="02_stim",
    segment_name=file_basename,
    record_start_epoch=segment_origin,
)

saver.start_recording([0])
audit_log.add(event_type="record_start", block=block, phase="02_stim",
              segment_name=file_basename, epoch_sec=segment_origin)

for i, pulse in enumerate(pulses, start=1):
    t_stim = time.time()
    seq.send()  # 或 sendSequence
    segment_log.add_stim(epoch_sec=t_stim, stim_index=i, extra={"mode": "burst_end"})
    audit_log.add(event_type="stim_send", block=block, phase="02_stim",
                  segment_name=file_basename, epoch_sec=t_stim,
                  extra={"stim_index": i})

saver.stop_file()
audit_log.add(event_type="record_end", block=block, phase="02_stim",
              segment_name=file_basename, epoch_sec=time.time())

# ★ 段结束立即写段内时间表（分析主文件）
segment_log.save_txt(phase_dir / "stim_times.txt")
segment_log.save_json(phase_dir / "segment_time_meta.json")  # 可选

# ---------- run 结束 ----------
audit_log.mark_experiment_end()
audit_log.save()  # -> run_dir/external_time_table.csv（可选审计）
```

闭环 C++ 触发刺激时：每次下发须回传 `epoch_sec` 给 Python，写入**当前 02_stim 段**的 `SegmentStimLog`，段结束统一 `save_txt`。

---

## 十、生成代码时 AI 必须输出的内容

1. **完整目录树**（含空 `cpp/build/.gitkeep` 若需要）  
2. **`README.md`**：实验目的、参数表、运行命令、环境变量、与 demo 差异  
3. **`main.py` + `python/experiment_runner.py`**：符合第五节、第六节、**第九节外界时间**  
4. **`python/utils/time_log.py`**（或等价模块）：每段 `02_stim` 写 `stim_times.txt`；run 结束可选写 `external_time_table.csv`  
5. **`config/*.yaml`**：可运行的示例值（路径用占位符并注释）  
6. **`requirements.txt` + 环境检查**  
7. **闭环实验**：C++ 源码 + `CMakeLists.txt` + Python 侧 subprocess 调用  
8. **不得**把 secrets、绝对路径硬编码进 git；默认值可写 demo 路径但须在 README 说明如何改 yaml  

---

## 十一、给 AI 的简短复制用 Prompt（精简版）

```text
请按 MaxWell 实验框架规范生成完整实验项目：

【实验信息】
（在此填写：细胞编号、DIV、日期、实验名、科学问题、闭环逻辑、记录 block 结构）

【必须遵守】
1. 目录：main.py 入口；config/system.yaml + stimulation.yaml；python/ + cpp/ + scripts/ + data/
2. 每次 main 运行创建 data/{YYYYMMDD_HHMMSS}_data/，内含：复制的 cfg、log.txt、config 快照；raw_data/ 下有多组 block（按实验条件命名），每组 block 内固定 01_pre_spont / 02_stim / 03_post_spont 三段
3. 刺激：电极组 + 三类协议（single_pulse / individual_burst / sequence_with_burst）；幅值用 query_DAC_lsb_mV() 换算
4. mx.Event properties 必须 key-value 成对，如 "type stim"
5. 闭环：Python 配置录制+Sequence，C++ DataStreamerFiltered 实时检测并 sendSequence；先编译 C++ 再开录
6. 启动时检查 requirements.txt，缺失写 log 与修复建议，齐全则 log「环境检查: OK」
7. README.md 写清实验计划与运行方式
8. **刺激时间**：每段 `02_stim` 生成 `stim_times.txt`（相对该段起点的秒，与 H5 同目录）；可选 run 级 external_time_table.csv 仅作审计

【参考实现】
0520test_closed_stim/（闭环）；28913_amp_gradient_0525/（开环 + 时间总表）

请生成可直接运行的代码，不要省略 main.py 与 config 示例。
```

---

## 十二、与现有 Demo 的对照

| 项目 | Demo 现状 | 框架目标 |
|------|-----------|----------|
| 入口 | `setup_test_closed_loop.py` | `main.py` |
| 配置 | 脚本内常量 + 环境变量 | `config/system.yaml` + `stimulation.yaml` |
| 数据目录 | `SCRIPT_DIR` | `{data_root}/{timestamp}_data/raw_data/...` |
| log | 仅 print | `run/log.txt` + console |
| cfg 归档 | 不复制 | 复制到 run 目录 |
| C++ 构建 | 脚本内 cmake | main 自动构建或 `scripts/build_cpp.sh` |
| events | `"type stim"` ✓ | 所有刺激协议统一加 Event |
| 外界时间 | 无 / 仅 block 级 json | 每段 **stim_times.txt** + 可选 run 级审计 CSV |

---

*文档版本：2026-05-27 · 段内 stim_times.txt 为主、run 级时间表为辅*
