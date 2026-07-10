# MaxWell 数据规范（Data Template）

> **适用范围**：`maxwell_platform` 后续所有实验与分析所依赖的**单次数据包**。  
> 本文描述的是**逻辑组成**（三类文件），而非某一次实验的固定路径或文件名。示例路径仅用于说明结构。

---

## 数据包总览

一次可分析的数据由 **三部分** 组成：

| 部分 | 是否必需 | 典型扩展名 / 命名 | 作用 |
|------|----------|-------------------|------|
| **① 数据主体** | 必需 | `*.raw.h5` | MaxWell 录制的 spike、事件、原始帧、通道映射与录制元数据 |
| **② 刺激时间文件** | **推荐**（`02_stim` 段） | `stim_times.txt`（同目录）；可选 run 级 `external_time_table.csv` | 段内秒数相对该 H5 起点；当 H5 `events` 缺失时用 `stim_times.txt` 对齐 Raster / batch |
| **③ 电极 map 文件** | **可选** | `*.cfg` | 实验所用电极布局（电极 ID、通道号、平面坐标），供空间分析与刺激电极定位 |

```text
{run 目录}/
├── external_time_table.csv     # ② 可选；run 级审计总表（勿单独用于段内对齐）
├── {HHhMMmSSs}.cfg             # ③ 电极 map
└── raw_data/{block}/02_stim/
    ├── *.raw.h5                # ① 数据主体
    └── stim_times.txt          # ② 推荐；与本 H5 一一对应（见 experiment_template 第九节）
```

**对齐原则**

- 分析对齐优先读 **② 同目录 `stim_times.txt`**（相对该 H5 段起点的秒）；其次 H5 `events`；run 级 CSV 的 `offset_sec` 仅作审计。
- 空间分析统一使用 **③ cfg** 的电极坐标；H5 内 `settings/mapping` 为录制当时通道—电极绑定，二者应一致，若冲突以 **③ cfg + 实验记录** 为准并记入 `notes`。

---

## 一、数据主体：`*.raw.h5`

### 1.1 文件来源与命名

- 由 MaxWell `mx.Saving()` 录制生成，常见后缀 **`.raw.h5`**。
- 命名因实验而异，推荐模式（见 `experiment_template.md` 第八节）：

  `{recording_name_prefix}_{block}_{phase}.raw.h5`

  例：`stim_1hz_10p.raw.h5`、`28913_amp_el_2101_02_stim_150mV_10p.raw.h5`。

- **一个 H5 通常对应一段连续录制**（一个 phase 或一次闭环段）；多 block / 多 phase 实验会有多个 H5。

### 1.2 顶层结构（根 Group）

根级键名在 MaxWell 25.x 录制文件中稳定出现，示例如下（参考文件：`stim_1hz_10p.raw.h5`）：

| 路径 | 类型 | 说明 |
|------|------|------|
| `hdf_version` | Dataset | HDF 库版本字符串 |
| `mxw_version` | Dataset | MaxWell 软件版本 |
| `version` | Dataset | 文件格式版本 |
| `wellplate/` | Group | 孔板信息：`id`（如细胞/板编号 `28937`）、`variant`、`version` |
| `wells/` | Group | 各 well 元数据 |
| `recordings/` | Group | 录制索引，通常含 `rec0000/` |
| **`data_store/`** | Group | **主要分析数据**，含 `data0000/`（多段录制时为 `data0001/` …） |
| `assay/` | Group | 实验/assay 输入描述 |
| `bits/` | Group | 位流相关（常为空） |
| `environment/` | Group | 温度、诊断等（可为空） |
| `notes/` | Group | 备注 |

**分析时优先使用 `data_store/dataXXXX/`**；`recordings/recXXXX/well000/` 为镜像/索引结构，字段与 `data_store` 侧对应段一致。

### 1.3 核心数据段：`data_store/data0000/`

以下路径相对于 `data_store/data0000/`（`data0000` 为第一段录制，多段时递增编号）。

#### 时间与 Well

| 数据集 | dtype | 说明 |
|--------|-------|------|
| `start_time` | int64 | 录制开始时间戳（**毫秒**，Unix epoch） |
| `stop_time` | int64 | 录制结束时间戳（毫秒） |
| `well_id` | int32 | Well 索引（单孔板常为 `0`） |
| `recording_id` | int32 | 录制 ID |

时长（秒）：

```text
duration_sec ≈ (stop_time - start_time) / 1000
```

可与 `groups/exp/frame_nos` 交叉验证（见 1.5 节）。

#### 录制设置：`settings/`

| 数据集 | 说明 | 示例值 |
|--------|------|--------|
| `sampling` | 采样率 (Hz) | `20000` |
| `gain` | 放大器增益 | `512` |
| `hpf` | 高通截止 (Hz) | `300` |
| `lsb` | ADC LSB (V) | `6.29e-6` |
| `spike_threshold` | 事件检测阈值 | 依实验配置 |
| **`mapping`** | 结构化数组，**全阵列**通道—电极—坐标 | 见下表 |

`mapping` 字段：

| 字段 | 类型 | 含义 |
|------|------|------|
| `channel` | int32 | 数据通道号（与 `spikes.channel`、`groups/exp/channels` 一致） |
| `electrode` | int32 | 电极 ID |
| `x`, `y` | float64 | 电极平面坐标 (µm) |

#### Spikes：`spikes`

| 字段 | 类型 | 含义 |
|------|------|------|
| `frameno` | int64 | 全局帧号（**非**从 0 起的相对帧，见 1.5） |
| `channel` | int32 | 通道号 |
| `amplitude` | float32 | 检测幅度 |

**读取示例（Python）**

```python
import h5py
import numpy as np

with h5py.File(h5_path, "r") as f:
    seg = f["data_store/data0000"]
    spikes = seg["spikes"][:]
    fs = float(seg["settings/sampling"][0])
    mapping = seg["settings/mapping"][:]
    # 相对时间（秒），以本段首帧为原点
    t0 = int(seg["groups/exp/frame_nos"][0])
    spike_times_sec = (spikes["frameno"] - t0) / fs
```

#### 刺激 / 标注事件：`events`

| 字段 | 类型 | 含义 |
|------|------|------|
| `frameno` | int64 | 事件所在帧 |
| `eventtype` | uint32 | 事件类型（MaxWell 定义，刺激相关常为 `1`） |
| `eventid` | uint32 | 事件 ID |
| `eventmessage` | bytes / str | 属性字符串，常为 JSON 或 key-value 文本 |

**示例**（梯度固定频率刺激，`stim_1hz_10p.raw.h5`）：

```text
frameno=247215863, eventtype=1, eventid=922000
eventmessage='{"grad_fixedfreq":"amp_mV=300","phase_us=200":"pulse=1/10"}'
```

- 共 9 条事件，相邻 `frameno` 间隔约 **1 s**，与文件名 `1hz` 一致。
- 若 `mx.Event` 的 `properties` 不是成对 key-value，事件可能**静默不写入** H5（见 `experiment_template.md` 5.3 节）——此时必须依赖 **第二节 刺激时间文件**。

#### 原始波形（可选加载）：`groups/exp/`

| 数据集 | shape | 说明 |
|--------|-------|------|
| `channels` | `(N_ch,)` | 本段写入的通道列表 |
| `frame_nos` | `(N_frames,)` | 每列对应的全局帧号 |
| `raw` | `(N_ch, N_frames)` uint16 | 原始 ADC 数据 |
| `triggered` | `(1,)` | 是否触发录制 |

- `N_frames / sampling ≈ duration_sec`（示例：179800 / 20000 ≈ 9.0 s）。
- 大数据量时分析 pipeline 可**只读 `spikes`**，按需再读 `raw`。

### 1.4 时间轴与 `frameno` 换算

`frameno` 为 **全局累计帧计数**，不是段内从 0 开始。

推荐相对时间（秒）：

```text
t_sec = (frameno - frame_nos[0]) / sampling
```

其中 `frame_nos[0]` 为本段 `groups/exp/frame_nos` 的最小值（或与 `start_time` 对齐的第一帧）。

与 **外界时间表** 对齐时：

1. 用 H5 的 `start_time`（ms）或 `record_start` 的 `epoch_sec` 作为段起点；
2. 刺激时刻：`t_stim_sec = (event_frameno - frame_nos[0]) / fs`；
3. 与 `external_time_table.csv` 中 `stim_send` 的 `offset_sec` 做差，偏差应在采样误差 + 时钟误差范围内；偏差过大时以 **外界时间表为主** 并记录原因。

### 1.5 平台读取接口约定（`io/`）

| 模块 | 职责 |
|------|------|
| `maxwell_reader.py` | 打开 H5、解析 `data_store/dataXXXX` 列表、暴露段元数据 |
| `spike_reader.py` | 读取 `spikes` 并返回标准化表（time, channel, amplitude, electrode） |
| `event_reader.py` | 解析 `events` / `eventmessage`（含 JSON） |
| `h5_reader.py` | 通用 HDF5 工具（遍历、缓存 mapping） |

**标准化 spike 表建议列**：`time_sec`, `channel`, `electrode`, `x`, `y`, `amplitude`, `frameno`（后处理可丢弃 `frameno`）。

### 1.6 数据质量检查清单

- [ ] `stop_time - start_time` 与预期 phase 时长一致  
- [ ] `len(spikes) > 0`（自发/刺激段视实验而定）  
- [ ] 刺激段：H5 `events` 条数是否与预期脉冲数一致；不一致则准备 **第二节 CSV**  
- [ ] `settings/mapping` 通道数与 `groups/exp/channels` 一致  
- [ ] `wellplate/id` 与实验记录（细胞编号）一致  

---


## 二、刺激时间文件（可选）

### 2.1 何时需要

在以下情况应提供 **`external_time_table.csv`**（或等价 JSON）：

| 情况 | 说明 |
|------|------|
| H5 无 `events` 或条数为 0 | 常见于 `mx.Event` properties 格式错误未写入 |
| 闭环由 **C++ `sendSequence`** 触发 | 刺激未经过 Python `mx.Event`，H5 内无对应标注 |
| 需要 **墙钟时间** 与多段 H5 / 外部设备对齐 | 跨文件、跨 block 对齐 |
| H5 事件时间需校验 | 用 `offset_sec` 与 `frameno/fs` 交叉验证 |

若 H5 `events` **完整且已校验**，本节文件可省略，但**仍建议**在 run 级保留一份总表以便追溯（见 `experiment_template.md` 第九节）。

### 2.2 文件位置与粒度

- **每个实验 run 一份总表**（非每个 H5 一份）：

  ```text
  {data_root}/{YYYYMMDD_HHMMSS}_data/external_time_table.csv
  ```

- 可选镜像：`external_time_table.json`（内容与 CSV 一致）。
- 文件名固定为 `external_time_table`；**不要**与具体 H5  basename 绑定。

### 2.3 CSV 列定义

**必需列**（与 `experiment_template.md` §9.4 一致）：

| 列名 | 说明 |
|------|------|
| `event_type` | `experiment_start` / `record_start` / `record_end` / `stim_send` / `experiment_end` |
| `block` | 实验 block 名；实验级事件可为空 |
| `phase` | 如 `01_pre_spont`、`02_stim`、`03_post_spont`、`recording`、`closed_loop_stim` |
| `segment_name` | 对应 H5 的 `start_file` basename（**无扩展名**），用于关联 ① |
| `wall_time_utc` | ISO8601 UTC |
| `wall_time_local` | 本机时区 ISO8601 |
| `epoch_sec` | `time.time()` 浮点秒 |
| `offset_sec` | 相对本次 run 的 `time_origin_epoch`（秒） |

**可选列**（按实验扩展）：

| 列名 | 说明 |
|------|------|
| `row_index` | 行号（便于人工查阅） |
| `pulse_index` | 刺激序号 |
| `amplitude_mv` | 刺激幅值 (mV) |
| `electrode` | 刺激电极 ID |
| `total_spikes` | 触发时累计 spike 数（闭环） |
| `note` | 如 `cpp_sendSequence`、协议名等 |

### 2.4 示例（节选）

参考结构（闭环实验，`external_time_table.csv`）：

```csv
row_index,event_type,block,phase,segment_name,wall_time_utc,wall_time_local,epoch_sec,offset_sec,pulse_index,amplitude_mv,electrode,total_spikes,note
1,experiment_start,,,,2026-05-25T10:51:33.600091Z,2026-05-25T18:51:33.600091+08:00,1779706293.6000912,0.0,,,,,
2,record_start,closed_loop,recording,closed_loop_test,2026-05-25T10:51:56.855152Z,...,1779706316.8551517,23.25506,,,,,
3,stim_send,closed_loop,closed_loop_stim,closed_loop_test,2026-05-25T10:51:59.378516Z,...,1779706319.3785164,25.778425,1,150.0,5265,301,cpp_sendSequence
...
13,record_end,closed_loop,recording,closed_loop_test,2026-05-25T10:52:01.391287Z,...,1779706321.3912873,27.791196,,,,,
14,experiment_end,,,,2026-05-25T10:52:01.392218Z,...,1779706321.3922179,27.792127,,,,,
```

### 2.5 与 H5 的关联方式

```text
segment_name  ==  H5 文件名去掉 ".raw.h5"
```

对齐步骤：

1. 取 `record_start` / `record_end` 的 `offset_sec` 界定该 H5 在 run 时间轴上的区间；  
2. 取 `stim_send` 行，用 `segment_name` 过滤属于当前 H5 的刺激；  
3. 计算 `t_h5_sec = offset_sec - offset_sec(record_start)`；  
4. 与 `(event_frameno - frame_nos[0]) / fs` 比较，记录最大偏差。

**平台读取**：`io/stim_reader.py` 或 `analysis/utils/stim/` 中实现 CSV/JSON 加载，输出统一 `StimEvent` 表（`time_sec`, `pulse_index`, `amplitude_mv`, `electrode`, `source`）。

---

## 三、电极 map 文件：`*.cfg`

### 3.1 文件来源与命名

- MaxWell 阵列配置文件，实验开始前由 `array.load_config(cfg_path)` 加载。
- 每次 run 应将**源 cfg 复制**到 run 目录并归档，命名通常为录制开始时刻：

  ```text
  {HHhMMmSSs}.cfg    # 例：18h51m33s.cfg
  ```

- **路径与文件名因实验而异**；分析时通过 run 目录或 `config/system.yaml` 的 `electrode_map.cfg_path` 定位，不要写死为某一固定文件名。

### 3.2 文本段格式（主体）

文件主体为分号分隔条目，每条描述一个**已选电极**：

```text
{channel}({electrode_id}){x}/{y};
```

| 片段 | 含义 | 示例 |
|------|------|------|
| `channel` | 数据通道号（与 H5 `spikes.channel` 一致） | `817` |
| `electrode_id` | 电极 ID（整数，括号内） | `773` |
| `x`, `y` | 平面坐标 (µm) | `2747.5`, `52.5` |

**示例条目**（参考 `03h16m40s.cfg` / `18h51m33s.cfg`）：

```text
0(16095)612.5/1277.5;
773(817)2747.5/52.5;
965(2101)2117.5/157.5;
```

阵列尺寸为 **220 列 × 120 行**（`MAP_SHAPE = (120, 220)` 行×列）；坐标换算 `col=round(x/17.5)`, `row=round(y/17.5)`，与 H5 `settings/mapping` 一致。解析后应与 H5 mapping **几乎全部格点重合**；仅刺激等特殊电极可能仅在 CFG 中有完整编号对应关系。

说明：

- 电极 ID **不要求连续**；常见为稀疏编号或奇数序列（与 MaxWell 阵列编号一致）。
- 同一电极可能出现**重复行**（如 `317(3503)...` 出现两次），解析时建议 **去重**（以 `electrode_id` 或 `(electrode_id, channel)` 为键，保留首次或末次并打 log）。
- 解析后条目数（示例约 **443** 条）为**选中电极子集**，不必等于 H5 `mapping` 全长（示例 H5 为 1020 通道全映射）。

### 3.3 二进制尾段（可选）

部分 cfg 在文本列表后附带 **Base64 编码的 gzip 数据**（常以 `H4sI` 开头）。平台解析策略：

1. 优先解析分号分隔文本段，满足绝大多数空间分析与刺激电极查询；  
2. 若存在 `H4sI` 尾段且文本段不完整，可解码 gzip 作为补充（实现放在 `io/` 或 `analysis/utils/io/`）；  
3. 解码失败时不阻塞流程，但应在 log 中警告。

### 3.4 与 H5 `settings/mapping` 的关系

| 来源 | 内容 | 用途 |
|------|------|------|
| **cfg（③）** | 实验设计时选中的电极 + 坐标 | 刺激电极、ROI、连接性分析默认布局 |
| **H5 mapping（①）** | 录制时系统保存的全通道映射 | 将任意 `spikes.channel` 转为 `(electrode, x, y)` |

流程建议：

```text
spikes.channel  --[H5 mapping]-->  electrode, x, y
刺激电极列表      --[cfg 解析]-->    electrode, x, y   （校验是否 ⊂ 选中集）
```

若同一 `channel` 在 cfg 与 H5 mapping 中 `electrode` 不一致，标记为 **mapping 冲突** 并人工核对。

### 3.5 平台解析输出约定

解析函数输出 **DataFrame / 表**，建议列：

| 列名 | 类型 | 说明 |
|------|------|------|
| `electrode` | int | 电极 ID |
| `channel` | int | 通道号 |
| `x`, `y` | float | µm |
| `selected` | bool | 是否在本次实验选中（cfg 即为 True） |

**读取模块**：`io/stim_reader.py` 旁可增加 `cfg_reader.py`，或置于 `analysis/utils/io/electrode_map.py`。

---

## 四、三类文件的组合使用

### 4.1 最小数据包（开环，H5 事件完整）

```text
run_dir/
├── group_A_02_stim.raw.h5
└── 12h30m00s.cfg
```

分析：读 H5 spikes + events；空间用 cfg；无需 CSV。

### 4.2 推荐完整数据包（闭环或多段）

```text
20260525_185133_data/
├── 18h51m33s.cfg
├── external_time_table.csv
└── raw_data/
    └── closed_loop/
        └── closed_loop_test.raw.h5   # segment_name = closed_loop_test
```

分析：CSV 提供 `stim_send`；H5 提供 spikes；cfg 提供电极布局；`segment_name` 串联 ① 与 ②。

### 4.3 分析 pipeline 加载顺序（建议）

```text
1. 加载 cfg  → 电极表 electrode_map
2. 打开 H5   → spikes, events, settings, frame_nos
3. 若 events 不足或实验为闭环 → 加载 external_time_table.csv
4. 合并刺激表：优先 CSV（若存在且 flag_use_external_stim=True），否则 H5 events
5. 将 spike 时间转为相对秒，挂载 electrode/x/y
```

---

## 五、引用示例（仅供对照结构）

以下路径为**示例**，后续数据不必同名，但应保持**相同逻辑结构**：

| 类型 | 示例路径 |
|------|----------|
| H5 | `G:\data_maxwell\...\stim_1hz_10p.raw.h5` |
| 刺激时间 CSV | `...\20260525_185133_data\external_time_table.csv` |
| 电极 cfg | `...\20260525_185133_data\18h51m33s.cfg` |

实验录制与 CSV 生成规范详见：`docs/experiment_template.md`（第八、九节）。

---

*文档版本：2026-05-27 · 依据 MaxWell 25.1.8.2 录制文件与闭环 run 示例整理*
