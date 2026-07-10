# analysis/utils 工具函数索引

`analysis/utils/` 存放**与具体实验无关**的可复用分析工具。  
`analysis/interactive_pipeline2/` 为独立 GUI 管线，其 utils 不在此目录，也不应与本目录混用。

**维护约定**：每次在 `analysis/utils/` 新增、修改或删除公开 API 时，须同步更新本文档对应章节。

---

## 目录结构概览

| 子包 | 用途 |
|------|------|
| `io/` | MaxWell H5 读写、电极 mapping、缓存 |
| `stim/` | 刺激诱发稳定响应电极、自发稳定延迟连接 |
| `map/` | 电极活动热图、GIF 导出 |
| `raster/` | Spike 序列构建、population rate、刺激伪迹过滤 |
| `detection/` | Burst 检测 |
| `connectivity/` | STTC、传递熵、相关合并、图构建 |
| `burst_phase/` | Burst 相位分析管线 |
| `plotting/` | Matplotlib 中文字体等通用绘图 |
| `gradient_batch/` | 幅值梯度 batch 分析管线（指标、图表、IO） |
| `precise_timing_batch/` | 精确时序实验 batch（baseline 连接、可塑性、FC） |
| `interactive/` | 与 GUI 共用的少量计算（单电极 raster 等） |

---

## stim/ — 稳定响应与稳定延迟

来源口径：`0619_28878_local_multi_amp_gradient` / `0623_26056` 等同 family batch 分析。

### 刺激诱发「稳定激活电极」（stable activated electrode）

对每个电极：每次刺激后在 `[tmin_ms, tmax_ms]` 取最早 spike 延迟 → 跨 pulse 做直方图 → `peak_ratio = max(bin) / median(nonzero bins)` → `ratio >= peak_ratio_thr` 判为 stable。

| 符号 | 说明 |
|------|------|
| 模块 | `analysis/utils/stim/stable_response.py` |
| `StableResponseConfig` | 参数 dataclass（默认 tmin=3, tmax=200, bin=1, thr=20, min_trials=3） |
| `StableResponseConfig.from_mapping(cfg)` | 从 batch `response_map` 配置段构造 |
| `peak_ratio_from_histogram_counts(counts)` | 直方图 peak ratio |
| `latency_histogram_edges(tmin_ms, tmax_ms, bin_ms)` | 延迟直方图 bin 边界 |
| `first_spike_latencies_ms_per_stim(ts_sec, stim_t_sec, ...)` | 单通道每次刺激的最早 spike 延迟 (ms) |
| `classify_channel_stable_response(latencies_ms, cfg)` | 单通道 stable 判定；非 stable 返回 `None` |
| `stable_electrode_metrics_from_stim(spike_t, spike_ch, stim_t, ch_to_ele, ele_layout, ...)` | 返回 stable 电极列表（含 ele/channel/peak_ms/peak_ratio/x/y） |
| `stable_electrode_set_from_stim(...)` | 返回 `set[int]` 电极 ID |

**示例**

```python
from analysis.utils.io.read_h5 import read_spikes, read_stim_times
from analysis.utils.stim import stable_electrode_metrics_from_stim, StableResponseConfig

spikes, paths, t0 = read_spikes(h5_path)
stim_t = read_stim_times(h5_path, paths=paths, frame_origin=t0)
rows = stable_electrode_metrics_from_stim(
    spikes.time_sec, spikes.channel, stim_t, ch_to_ele, ele_layout,
    map_cfg={"tmin_ms": 3.0, "tmax_ms": 200.0, "peak_ratio_thr": 20.0},
)
stable_set = {r["ele"] for r in rows}
```

### 自发 burst「稳定延迟连接」（stable delay directed pairs）

对每个 burst 取通道首次激活延迟 → 通道对 delay 直方图 peak ratio → 有向连接边。

| 符号 | 说明 |
|------|------|
| 模块 | `analysis/utils/stim/stable_delay.py` |
| `StableDelayPairConfig` | pair_bin_ms、pair_peak_ratio_thr、pair_min_shared_bursts 等 |
| `first_activation_ms_per_burst(t_sec, bursts)` | burst 内首次 spike 延迟 (ms) |
| `peak_ratio_from_values(vals_ms, bin_ms)` | 无固定上界的 delay 序列 peak ratio |
| `stable_delay_directed_pairs(spike_lookup, bursts, ch_to_ele, cfg)` | 返回 `(edges, out_degree)` |

---

## io/ — H5 与电极布局

| 模块 | 主要 API |
|------|----------|
| `read_h5.py` | `read_spikes`, `read_stim_times`, `read_recording_info`, `get_frame_origin`, `build_channel_map_matrix`, `extract_spike_waveforms`, `append_manual_stim_events`, `diagnose_h5` |
| `electrode_map.py` | cfg/H5 电极坐标解析 |
| `h5_cache.py` | H5 读取缓存 |
| `maxwell_hdf5_plugin.py` | mxw 压缩 raw 插件注册 |

**约定**：刺激时间优先从 H5 `events` 读取（见 `read_stim_times`）；`read_spikes` 与 `read_stim_times` 默认共用 `get_frame_origin` 时间基准。

---

## map/ — 电极活动图

| 模块 | 主要 API |
|------|----------|
| `electrode_activity.py` | `compute_channel_metrics`, `build_playback_frames`, `metrics_to_display_grid`, `MapFrame` |
| `export_gif.py` | `export_map_sequence_gif` |

---

## raster/ — Spike 序列

| 模块 | 主要 API |
|------|----------|
| `series.py` | `build_raster_series_from_spikes`, `population_rate_trace`, `filter_series_exclude_stim_artifacts`, `counts_on_channel_map` |

---

## detection/ — Burst

| 模块 | 主要 API |
|------|----------|
| `burst.py` | `detect_burst_intervals`（池化 ISI / 阈值法，与 batch 配置字段对齐） |

---

## connectivity/ — 连接性

| 模块 | 主要 API |
|------|----------|
| `sttc.py` | `sttc_pair`, `sttc_matrix` |
| `transfer_entropy.py` | `discrete_te_binary`, `te_matrix`, `spikes_to_binary_bins` |
| `electrode_merge.py` | `merge_correlated_electrodes`, `pearson_correlation_matrix` |
| `pipeline.py` | `compute_connectivity_graph`, `ConnectivityGraph`, `sample_burst_intervals` |

---

## burst_phase/

Burst 相位提取与指标、出图管线（`pipeline.py`, `phase.py`, `metrics.py`, `figures.py`）。

---

## plotting/

| 模块 | 主要 API |
|------|----------|
| `matplotlib_zh.py` | `configure_matplotlib_chinese`, `chinese_font_properties` |

---

## gradient_batch/

幅值梯度类 batch 分析一站式工具：H5 IO、burst FR、刺激事件、指标核心、表格管线、统一出图风格等。入口参考 `gradient_batch/run_analysis.py`。

| 模块 | 用途 |
|------|------|
| `metrics_core.py` | 自发/刺激响应核心指标 |
| `stim_events.py` | 刺激事件解析 |
| `burst_fr.py` | Burst 发放率 |
| `table_pipeline.py` | 表格汇总 |
| `figure_style.py` | 统一样式 |
| `pipeline_logging.py` | 日志与进度条配置 |

---

## precise_timing_batch/

精确时序 / baseline 窗口类实验 batch 工具。

| 模块 | 主要 API |
|------|----------|
| `io.py` | `discover_trials`, `load_spike_data`, `load_stim_times` |
| `baseline_connectivity.py` | `run_baseline_connectivity`, `first_activation_matrix`, `BaselineConnectivityConfig` |
| `plasticity.py` | 延迟漂移统计、配对检验 |
| `evoked.py` | 诱发响应计数与条形图 |
| `burst_dynamics.py` | PSTH、burst 特征表 |
| `firing_rate.py` | 电极发放率热图 |
| `fc_remodeling.py` | STTC 三元组、FC 矩阵 |

> 注意：`baseline_connectivity.py` 与 `stim/stable_delay.py` 均处理 burst 对齐 delay，但前者面向 precise_timing 完整 map 管线，后者为 0619 系列轻量 API。新 batch 优先使用 `stim/stable_delay.py`；需要 pre-exp 风格箭头 map 时用 `baseline_connectivity`。

---

## interactive/

| 模块 | 主要 API |
|------|----------|
| `single_electrode.py` | `burst_aligned_raster_points`, `first_activation_density` |

---

## 变更记录

| 日期 | 变更 |
|------|------|
| 2026-06-30 | 新增 `stim/stable_response.py`、`stim/stable_delay.py`；`0619_28878` 改为引用公用 API |
