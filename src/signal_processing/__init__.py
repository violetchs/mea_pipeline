"""Signal processing module for feature extraction and analysis."""

import numpy as np
from scipy import signal

__all__ = ["SignalProcessor"]


class SignalProcessor:
    """Process MEA signals."""
    
    def __init__(self, sampling_rate=10000.0, filter_type="bandpass", freq_range=(300, 3000)):
        """Initialize SignalProcessor."""
        self.sampling_rate = sampling_rate
        self.filter_type = filter_type
        self.freq_range = freq_range
    
    def process(self, data):
        """
        Process MEA signal data.
        
        Args:
            data: Preprocessed data array
            
        Returns:
            Processed data
        """
        filtered = self.apply_filter(
            data,
            filter_type=self.filter_type,
            freq_range=self.freq_range,
        )
        return filtered
    
    def extract_features(self, data):
        """Extract features from signal."""
        array = np.asarray(data, dtype=float)
        return {
            "rms": np.sqrt(np.mean(np.square(array), axis=1)),
            "peak_to_peak": np.ptp(array, axis=1),
            "mean_abs": np.mean(np.abs(array), axis=1),
        }
    
    def apply_filter(self, data, filter_type='bandpass', freq_range=(300, 3000)):
        """Apply frequency filter to signal."""
        array = np.asarray(data, dtype=float)
        if array.shape[-1] < 16:
            return array.copy()

        nyquist = self.sampling_rate / 2.0
        if filter_type == "none":
            return array.copy()
        if filter_type == "bandpass":
            if not 0 < freq_range[0] < freq_range[1] < nyquist:
                raise ValueError("Bandpass frequencies must satisfy 0 < low < high < Nyquist.")
            cutoff = [freq_range[0] / nyquist, freq_range[1] / nyquist]
            btype = "bandpass"
        elif filter_type == "highpass":
            if not 0 < freq_range[0] < nyquist:
                raise ValueError("Highpass frequency must satisfy 0 < low < Nyquist.")
            cutoff = freq_range[0] / nyquist
            btype = "highpass"
        elif filter_type == "lowpass":
            if not 0 < freq_range[1] < nyquist:
                raise ValueError("Lowpass frequency must satisfy 0 < high < Nyquist.")
            cutoff = freq_range[1] / nyquist
            btype = "lowpass"
        else:
            raise ValueError(f"Unsupported filter type: {filter_type}")

        b, a = signal.butter(3, cutoff, btype=btype)
        return signal.filtfilt(b, a, array, axis=-1)
    
    def detect_spikes(self, data, threshold=4):
        """Detect spike events in signal."""
        array = np.asarray(data, dtype=float)
        noise = np.std(array, axis=1, keepdims=True)
        cutoff = abs(threshold) * np.where(noise == 0, 1.0, noise).ravel()
        spikes = []
        for index, channel in enumerate(array):
            peaks, _ = signal.find_peaks(-channel, height=cutoff[index])
            spikes.append(peaks)
        return spikes
