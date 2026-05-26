"""High-level MEA processing pipeline orchestration."""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np

try:
    from .analysis import Analyzer
    from .mea_io import MEAReader, MEAWriter
    from .preprocessing import Preprocessor
    from .signal_processing import SignalProcessor
except ImportError:  # Support running modules from an editable src path.
    from analysis import Analyzer
    from mea_io import MEAReader, MEAWriter
    from preprocessing import Preprocessor
    from signal_processing import SignalProcessor


ProgressCallback = Optional[Callable[[int, str], None]]


@dataclass
class PipelineConfig:
    """User-configurable processing parameters."""

    sampling_rate: float = 10000.0
    filter_type: str = "bandpass"
    low_cut: float = 300.0
    high_cut: float = 3000.0
    outlier_threshold: float = 5.0
    normalize: bool = True
    spike_threshold: float = 4.0
    output_dir: str = "data/processed"


@dataclass
class PipelineResult:
    """Container for all pipeline stage outputs."""

    raw: np.ndarray
    preprocessed: np.ndarray
    processed: np.ndarray
    signal_features: Dict[str, np.ndarray]
    analysis: Dict[str, object]
    spikes: list
    output_path: Optional[Path] = None


class MEAPipeline:
    """Run the MEA workflow from file loading through analysis."""

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()

    def run(self, input_path: str, progress: ProgressCallback = None) -> PipelineResult:
        self._emit(progress, 5, "Loading MEA data")
        raw = MEAReader(input_path).load_data()

        self._emit(progress, 25, "Preprocessing data")
        preprocessor = Preprocessor(
            outlier_threshold=self.config.outlier_threshold,
            normalize_data=self.config.normalize,
        )
        preprocessed = preprocessor.preprocess(raw)

        self._emit(progress, 50, "Filtering signal")
        processor = SignalProcessor(
            sampling_rate=self.config.sampling_rate,
            filter_type=self.config.filter_type,
            freq_range=(self.config.low_cut, self.config.high_cut),
        )
        processed = processor.process(preprocessed)

        self._emit(progress, 68, "Extracting spikes and signal features")
        signal_features = processor.extract_features(processed)
        spikes = processor.detect_spikes(processed, threshold=self.config.spike_threshold)

        self._emit(progress, 82, "Running channel analysis")
        analysis = Analyzer().extract_features(processed)

        self._emit(progress, 92, "Saving processed data")
        output_path = self._save_processed(input_path, processed)

        self._emit(progress, 100, "Pipeline complete")
        return PipelineResult(
            raw=raw,
            preprocessed=preprocessed,
            processed=processed,
            signal_features=signal_features,
            analysis=analysis,
            spikes=spikes,
            output_path=output_path,
        )

    def _save_processed(self, input_path: str, data: np.ndarray) -> Path:
        source = Path(input_path)
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{source.stem}_processed.npy"
        MEAWriter(output_path).save_data(data)
        return output_path

    @staticmethod
    def _emit(progress: ProgressCallback, percent: int, message: str) -> None:
        if progress:
            progress(percent, message)
