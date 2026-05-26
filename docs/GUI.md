# MEA Pipeline GUI

The desktop GUI uses PySide6. It was selected because this project needs a
polished native interface, multiple result windows, and background worker
threads so long-running pipeline steps do not block the UI.

Install dependencies and start the app:

```powershell
python -m pip install -r requirements.txt
python -m src.gui.app
```

The GUI supports opening `.npy`, `.npz`, `.csv`, `.txt`, and `.tsv` MEA arrays,
previewing raw traces, editing processing parameters, running the full pipeline,
and opening separate result windows for summaries, correlation heatmaps, and
spike raster plots.

Blackrock `.nev` files can also be opened directly. NEV files are treated as
spike-event data: the GUI shows a channel/spike summary and raster plot. The
continuous-signal pipeline button is disabled for NEV files because those files
do not contain the raw voltage matrix expected by the current pipeline.
The NEV raster preview is interactive: the x-axis is time in seconds, the
y-axis uses the actual channel labels from the file, the slider moves through
the recording, and the controls adjust the visible time window as a number of
grid cells plus the grid resolution in milliseconds. For example, `50 grids`
and `100 ms/grid` shows a 5 second time window.
The mouse wheel zooms the raster around the cursor position: wheel up reduces
`ms/grid`, wheel down increases it. Wheel zoom uses ten times the step of the
spin controls for faster navigation.
Drag left or right on the raster to pan through time. The `Window` and `Grid`
fields keep plain text editing, with external `-` and `+` buttons beside each
field for step changes.
Click a channel row in the raster to inspect that channel. The selected channel
is highlighted in red in the raster, and its spike waveforms are drawn in the
waveform panel below the raster. The waveform panel samples large channels down
to a bounded number of traces and overlays the mean waveform for fast updates.

Sorting is available from `Tools > Sorting` or the sidebar. The sorting
workspace combines automatic and manual waveform sorting. The left plot shows
the selected channel's spike waveforms grouped by cluster, and the right plot
shows the reduction-space cluster distribution. The parameter panel supports
multiple reduction methods (`PCA`, `ICA`, or scaled waveforms) and multiple
clustering methods (`KMeans`, Gaussian mixture, agglomerative clustering, or
DBSCAN); it updates to show only the controls relevant to the selected methods.
Run auto sorting to initialize clusters for every channel, then select a channel
and use the lasso tool on the reduction plot to reassign spikes to a chosen
cluster. Manual lasso assignments can be reverted one step at a time with the
workspace `Undo` button.
`Assign Cluster` and `Assign Noise` work as persistent assignment modes: click
once to enter the mode, draw as many lasso regions as needed, then click the
same button again or right-click in the reduction plot to exit.
Use `Save Sorting` to export the current sorted data as a unified `.npz` file.
The file reuses the standard `UnifiedMEAData` layout: per-channel spike times,
waveforms, labels/embeddings, and additional per-channel/per-unit arrays such as
`unit_spikes_chan1_unit0` or `unit_waveforms_chan1_noise1`.

Channel maps are available from the sidebar or `Tools > Channel Map`. The map
editor shows an 8 x 8 MEA layout with circular electrodes. Green electrodes have
a channel assignment, red electrodes are unassigned, and reference electrodes
are marked with a black cross. Maps can be created from a blank layout, saved,
and saved as the default map. Saved maps are stored in
`config/channel_maps.json`.
