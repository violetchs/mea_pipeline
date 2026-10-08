# MEA Pipeline Chat Context and Change Log

Last updated: 2026-09-29

This document is a handoff note for future chats and agents. It records the main work requested in this conversation, the design decisions that must be preserved, recurring failure modes, and the current state of the latest prediction feature work. It is a context document, not a replacement for the source code or the tests.

## 1. Project Scope

The repository is a PySide6-based MEA pipeline for Maxwell/MaxOne recordings. The project contains:

- raw and processed data loading, file databases, raster and heatmap viewers;
- dynamic, stimulus-response, stable-delay and custom analysis workflows;
- Maxwell channel-map visualizations and stimulation-site selection;
- stimulus protocol/site/event-group/block generation and generated experiment packages;
- open-loop Maxwell hardware execution and a separate closed-loop module;
- an Agent Custom Code system for generating and running user-specific analysis modules;
- language selection and GUI localization.

The user expects implementation, not only a plan. Preserve existing user changes in the dirty worktree and make narrow changes in the responsible module.

## 2. Important Working Rules

### Channel IDs and maps

- Maxwell electrode IDs are real zero-based IDs. `row=0, col=0` is electrode `0`.
- Do not add the historical `+221` offset. It was determined to be an erroneous legacy operation.
- CFG/H5 electrode fields are authoritative. Grid coordinates are spatial metadata and must not silently renumber the electrode.
- Candidate stimulation electrodes must be selected using actual 2D map coordinates, not numerical proximity of electrode IDs.
- Non-heatmap channel-map drawings follow the Maxwell channel-map skill: white background, black electrode-area boundary, equally spaced gray electrodes, green recording electrodes, red stimulation electrodes, clickable electrode information, wheel zoom and mouse pan, with correct coordinate transforms.

### GUI/database behavior

- Named protocol, site, event-group, block-phase, block, processed-data and result records must use duplicate-name checks.
- A duplicate save must warn the user and must not overwrite the existing record.
- Loading an existing configuration should make it easy to create a new copy with an automatic short unique name.
- Do not reintroduce deleted Update/Save controls without an explicit request.
- Progress dialogs must show real progress or a clear indeterminate state; avoid a static empty progress bar that looks frozen.
- Long data-loading operations must not block the GUI thread unnecessarily.

### Testing and edits

- Preserve unrelated dirty changes in the worktree.
- Use focused tests and `python -m py_compile` for touched Python modules.
- The current repository contains many generated experiment folders and unrelated modified files. Do not delete or revert them unless explicitly requested.

## 3. Stimulus Generation: Implemented Direction

The stimulus-generation GUI was reorganized around a workflow of settings, protocol/site/event-group configuration, preview, block-phase creation, block creation and code generation.

### Protocols

Supported families include single pulse, individual burst, sequence burst, random burst and Poisson/randomized timing variants. Recent requested protocol behavior includes:

- `pulse_width_us` default: `200 us`;
- `pulse_frequency_hz` default: `20 Hz`;
- `start_ms` default: `1500 ms`;
- DAC channel default: implicit `0`, not shown as a normal editable field;
- random burst minimum interval default: `100 ms`;
- Poisson/random and uniform timing parameters are shown conditionally;
- random timing options show only the parameters for the selected distribution;
- the complete protocol duration is recorded after random sequence generation;
- default names should be short, descriptive and no longer than about 20 characters;
- protocol/site/event-group names must be generated without awkward `X`, `X_2`, `X_2_2` chains.

### Event groups and site switching

The intended model is:

1. A site group stores a center electrode and whether multi-electrode stimulation is requested, including the requested count.
2. At runtime, the loaded CFG is scanned for stimulation units.
3. For a multi-electrode site group, runtime selection chooses electrodes closest to the center, subject to connectability, unique stimulation units and recording-quality/rate constraints when available.
4. An event group can use one site group for all protocol events, or combine several site groups.
5. Event-group selection can be sequential or random, while preserving the required count for every subgroup in a block.
6. For burst protocols, the switch is per burst/event, not per pulse inside the burst. A two-pulse burst must keep both pulses on the same selected site group.
7. Event groups are selected from a library/checkable list, not a text field, and subgroup order is editable.

The GUI should update event subgroups immediately when site groups are checked, without requiring an Add-and-reload cycle.

### Hardware mapping and DAC lessons

The Maxwell API distinction matters:

- `connect` establishes the electrode-to-stimulation-unit route;
- `power_up` prepares stimulation units/power circuitry;
- DAC configuration determines signal and neutral levels;
- switching only routes while leaving the electrical state in an unsuitable configuration can produce weak or missing stimulation.

External reference experiments showed that event-level switching should not power up every event. The pipeline should keep a stable hardware configuration when possible, route only the units needed for the current event, and use separate signal/neutral DAC handling where required by the hardware setup. Hardware route logs and unit mappings should be saved with each generated experiment.

Known hardware constraints and failure messages:

- MaxOne supports at most 32 stimulation units per route. Do not build a runtime union containing hundreds of electrodes just because an event group contains several alternatives.
- A single stimulation unit cannot be assigned conflicting DAC sources.
- A requested electrode may be present in the recording map but have no usable stimulation unit in the loaded CFG.
- The diagnostic path `stimulation_connection_diagnostics.json` and `stimulation_unit_probe.json` are the primary evidence for distinguishing CFG absence, missing unit, duplicate unit mapping and unit occupation conflicts.

When diagnosing a hardware failure, scan the actual loaded CFG, log every requested electrode, returned unit, route, DAC source, duplicate unit and replacement candidate, then stop before stimulation if running in diagnostic mode.

### CFG and sidecar data

- Generated experiment packages copy the CFG and save a config snapshot.
- Stimulation metadata may be distributed across H5, NPZ, TXT and JSON files. Matching must be folder-local and based on the exact recording/stimulation file, not a broad multi-directory search that can mix blocks.
- H5 files are not retroactively modified just to add stimulation metadata. When saving NPZ spike data, preserve aligned stimulus times and per-stimulus site metadata.
- For external datasets, locate the corresponding sidecar files in the same recording folder or its exact experiment-level metadata directory.

## 4. Stimulus-Response Analysis

The stimulus-response workflow was repeatedly refined to handle:

- one or multiple stimulus groups;
- repeated blocks with the same site pattern;
- grouping trials by actual stimulation-site identity rather than mixing all trials;
- removal of duplicate/overlapping trials when multiple pulses occur in a trial;
- stimulation-site marks on raster and latency maps;
- artifact counting after the stimulus marker, including configurable settling/artifact windows;
- stable latency electrode selection and latency distributions;
- global response PSTH, local/global response classification and trial selection.

The analysis must use exact stimulus timing and site metadata aligned to the selected file. A self-spontaneous folder usually contains only H5; a stimulation folder generally has H5 plus matching TXT/JSON/NPZ sidecars. Do not accidentally combine all stimulation sites from neighboring directories.

The selected-electrode latency distribution should show the number of trials represented in the legend, not only in the title. Latency maps should show stimulation sites clearly, support wheel zoom and mouse pan, and only draw the requested stable/selected electrodes where the UI says so.

## 5. Stable Delay and Channel Maps

Stable-delay map work added configurable parameters, stable electrode filtering, first-spike latency distributions and directed connection display. The map layout requirements are:

- electrode points equally spaced vertically and horizontally;
- white background and a tight black electrode-area frame;
- no large blank margins around the frame;
- the electrode map should receive the main panel area rather than being squeezed by side plots;
- selected electrodes must visibly highlight;
- clicking a displayed electrode must map to the actual electrode under the cursor;
- wheel zoom must preserve the axes transform and never make the map disappear;
- drag panning must work without changing the data coordinates.

The stable activation peak plot was changed from all-electrode peak distribution to selected-electrode first-time latency distribution. The latency x-axis was limited to 200 ms in that view.

## 6. Raster/Heatmap Performance and Interaction

Raster requirements accumulated during the project:

- open raster should show progress while loading large files;
- burst detection and stimulus-time markers must remain visible;
- marker labels should not cover raster rows or global response plots;
- the raster time window must support wheel zoom and drag panning;
- channel selection must be fast and synchronized with the channel map;
- heatmap playback and panning should use cached/precomputed data and avoid full redraws when only a view window changes;
- waveform extraction may skip oversized channels when the load budget is exceeded, but the UI must explain that a spike train can exist even when waveform extraction is unavailable.

Common warning to treat as non-fatal unless it indicates a real rendering problem:

`MatplotlibDeprecationWarning: get_cmap` should use `matplotlib.colormaps` in new code.

## 7. Custom Analysis and Agent Custom Code

Custom analysis was changed from a fixed dimensionality-reduction entry point into a workflow where users can select files, time ranges and channels, run basic analyses such as vector firing rate, store processed data, then plot selected x/y data. Multiple input files and processed-data subsets are supported.

The Agent Custom Code system is intentionally not a git worktree and must not modify fixed `src/` code. It stores runtime artifacts under `.agents/`:

- `.agents/generated_modules/<module_id>/manifest.json`
- `.agents/generated_modules/<module_id>/module.py`
- `.agents/generated_modules/<module_id>/README.md`
- `.agents/runs/<run_id>/input/`
- `.agents/runs/<run_id>/output/`
- `.agents/runs/<run_id>/agent.log`

Generated modules implement `analyze(context: dict) -> dict` and execute in a subprocess. The module receives an input manifest and writes outputs only into its run output directory. Output matrices must be finite two-dimensional numeric arrays. A result is not saved to the processed database if the name already exists or the output contains NaN/infinite values.

### Agent environment checks

The intended UX is a one-time preflight screen before opening the full Agent window:

1. Check the selected agent executable.
2. Check that the agent can receive the full task text.
3. Check that it can write a minimal module.
4. Check that the module can run and return a valid result.
5. Show green check marks or a red stop state; halt on the first failed step and guide repair.

After a successful check, reopening the Agent UI should show the completed state and allow Continue without repeating the test. The in-window operational UI should not contain duplicate environment-test controls.

### Agent pitfalls

- The local Codex CLI rejected `--ask-for-approval` in `codex exec`; use the CLI version's supported `--sandbox` options.
- A generated module that receives only `TASK - write code now` will ask for clarification or infer from directory files. The full user task must be inserted into the agent prompt explicitly.
- The module shell can be created successfully while `module.py` is only a placeholder. The final runtime check must inspect the actual file and execute `analyze`, not merely check process exit code.
- UTF-8 BOM in `module.py` causes `invalid non-printable character U+FEFF`; write generated Python as UTF-8 without BOM.
- Agent output with NaN/infinite matrix values must be rejected before database save.
- Result history is persistent within the session and should be grouped by module/input dataset. Unsaved temporary results are deleted when the app closes; explicitly saved results remain.
- The result preview should support one large figure with wheel zoom and drag pan rather than several small blank preview panels.

## 8. Maxwell Runtime Environment Pitfalls

### MaxLab Python

`ModuleNotFoundError: No module named 'maxlab'` means the running Python interpreter cannot see the MaxLab package. `MAXLAB_PYTHON_PATH` should point to the MaxLab site-packages directory, but it must be set for the interpreter actually running the experiment.

The MaxLab package may contain a stale shebang such as `/usr/bin/python`; invoking `__init__.py` as a shell command is incorrect. Import it through the intended Python environment. Mixing Python 3.10 MaxLab binary packages with Python 3.13 pipeline packages caused ABI errors such as:

- `kiwisolver._cext` missing;
- `numexpr: _ARRAY_API not found`;
- NumPy 1.x compiled extension used with NumPy 2.x.

Use a compatible environment for MaxLab and a separate GUI environment if necessary. Do not add MaxLab's Python site-packages globally to a mismatched Python interpreter.

### C++ closed-loop linking

Undefined references to `std::__cxx11::basic_string::_M_replace_cold` when linking `libmaxlab.a` indicate an ABI/compiler mismatch between the MaxLab static library and the selected GCC/libstdc++ toolchain. Changing only the compiler version may not solve it; the C++ ABI and library build toolchain must match the MaxLab distribution. The GUI/closed-loop Python environment is separate from the open-loop generated experiment environment.

## 9. Closed Loop

Closed-loop code should live outside `generated_visual_experiment`, which is reserved for open-loop generated packages. The closed-loop module should be callable from the main GUI and expose:

- live raster and heatmap on the left;
- a compact stimulation-state view in the middle;
- a reserved external-controller/network panel on the right;
- automatic channel selection that can change over time;
- configurable detection channels, response electrodes, thresholds and windows as groups, not single fixed electrodes;
- schedule-end and manual-close automatic saving, including H5 when configured.

Live rendering must not delay the stimulation latency path. Acquisition, stimulation decisions, file saving and display refresh should be decoupled.

## 10. Current Prediction Module

The current class is `StimulusPreStatePredictionWindow` in `src/gui/app.py`.

### Target variables

For global-response trials, the response PSTH uses a 100 ms post-stimulus window and 5 ms bins. It extracts:

- response peak;
- peak time;
- response duration.

### Input variables

The current input contains only:

- one selectable 50 ms-binned global firing-rate vector:
  - full pre window (1000 ms -> 20 values);
  - last 500 ms -> 10 values;
  - last 200 ms -> 4 values;
  - last 100 ms -> 2 values;
  - last 50 ms -> 1 value;
- `Previous burst gap`: the interval from the end of the previous burst to the current stimulus time, in ms.

If no earlier burst is available in metadata, the feature is encoded as 1000 ms, meaning the previous burst is outside the current analysis window.

Different firing-rate windows are mutually exclusive within one feature combination. A combination may contain one rate vector alone or one rate vector plus previous-burst gap. Duplicate result rows are collapsed so only the best model per logical feature combination is shown.

### Standardization

Each numeric input column is standardized by a `StandardScaler` inside the model pipeline. For every five-fold split, means and standard deviations are calculated only on that fold's training data and then applied to the validation data. The target variables remain in original units, so RMSE is reported in Hz or ms.

### Models and metrics

Models currently evaluated:

- Ridge linear;
- random forest;
- gradient boosting.

The displayed result is the best model per feature combination, selected by mean RMSE relative to the random baseline.

The random baseline now breaks the trial correspondence between input state and response:

- randomly permute `y` five independent times;
- for each permutation, run the same five-fold cross-validation;
- train the selected model on the shuffled response pairing;
- evaluate against the original validation responses;
- aggregate the resulting 25 validation evaluations as RMSE.

This replaced the older train-set response resampling baseline.

### Actual/predicted comparison

The prediction window now has a `Compare actual / predicted features` button. After selecting a result row, it reconstructs the same five-fold fold-held-out predictions and opens a figure with three panels:

- Peak;
- Peak time;
- Duration.

Black curves are actual trial features, red curves are predicted features, and each title reports the corresponding RMSE. The figure uses global-response trial order on the x-axis.

## 11. Recent Verification

The following checks were run after the latest prediction changes:

```text
python -m py_compile src/gui/app.py tests/test_gui.py
4 selected pre_state tests passed
```

The focused tests cover:

- unique feature combinations and mutually exclusive rate windows;
- one displayed best model per feature combination;
- 50 ms rate-vector dimensions and values;
- five shuffled-pairing random baseline with five folds.

## 12. Recommended Next Steps

1. Add a GUI smoke test that instantiates `StimulusPreStatePredictionWindow`, selects a result row and opens the comparison dialog in offscreen mode.
2. Add a small synthetic record test for `_prepare_dataset` with explicit `trial_previous_burst_gap_ms` and verify the target/feature row alignment.
3. Replace remaining hard-coded English labels in the new prediction comparison controls with the project localization mechanism.
4. Consider caching fold-held-out predictions so clicking Compare does not retrain the selected model.
5. Keep the random baseline seed explicit and display the seed/permutation count in the summary for reproducibility.
6. Before hardware experiments, run CFG stimulation-unit diagnostics and inspect the saved route/DAC log.

## 13. Do Not Reintroduce These Known Problems

- Do not add `+221` to electrode IDs.
- Do not mix stimulation sidecars from neighboring directories.
- Do not treat a successfully created Agent module shell as proof that the task was understood or executable.
- Do not run generated Agent modules in the GUI main process.
- Do not compare a model's training predictions to actual responses when reporting prediction performance.
- Do not use MSE labels after the prediction UI has been converted to RMSE.
- Do not place multiple firing-rate windows in one feature combination.
- Do not route a large union of all possible event-group electrodes when the hardware limit is 32 stimulation units per route.
