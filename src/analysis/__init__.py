"""Analysis module for statistical and feature analysis."""

import numpy as np

__all__ = ["Analyzer"]


class Analyzer:
    """Analyze MEA data."""
    
    def __init__(self):
        """Initialize Analyzer."""
        pass
    
    def extract_features(self, data):
        """
        Extract neural features from processed data.
        
        Args:
            data: Processed signal data
            
        Returns:
            Feature matrix
        """
        stats = self.compute_statistics(data)
        correlations = self.correlate_channels(data)
        return {
            "statistics": stats,
            "correlation": correlations,
            "channel_count": int(np.asarray(data).shape[0]),
            "sample_count": int(np.asarray(data).shape[-1]),
        }
    
    def compute_statistics(self, data):
        """Compute statistical measures."""
        array = np.asarray(data, dtype=float)
        return {
            "mean": np.mean(array, axis=1),
            "std": np.std(array, axis=1),
            "min": np.min(array, axis=1),
            "max": np.max(array, axis=1),
            "rms": np.sqrt(np.mean(np.square(array), axis=1)),
        }
    
    def correlate_channels(self, data):
        """Compute cross-correlation between channels."""
        array = np.asarray(data, dtype=float)
        if array.shape[0] < 2:
            return np.ones((array.shape[0], array.shape[0]))
        return np.nan_to_num(np.corrcoef(array))
