"""Preprocessing module for MEA data cleaning and normalization."""

import numpy as np

__all__ = ["Preprocessor"]


class Preprocessor:
    """Preprocess MEA data."""
    
    def __init__(self, outlier_threshold=5.0, normalize_data=True):
        """Initialize Preprocessor."""
        self.outlier_threshold = outlier_threshold
        self.normalize_data = normalize_data
    
    def preprocess(self, data):
        """
        Preprocess raw MEA data.
        
        Args:
            data: Raw data array
            
        Returns:
            Preprocessed data
        """
        array = np.asarray(data, dtype=float)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        array = np.nan_to_num(array, copy=True)
        array = self.filter_outliers(array, threshold=self.outlier_threshold)
        if self.normalize_data:
            array = self.normalize(array)
        return array
    
    def normalize(self, data):
        """Normalize data to standard range."""
        array = np.asarray(data, dtype=float)
        mean = np.mean(array, axis=1, keepdims=True)
        std = np.std(array, axis=1, keepdims=True)
        return (array - mean) / np.where(std == 0, 1.0, std)
    
    def filter_outliers(self, data, threshold=3):
        """Remove outliers using statistical methods."""
        array = np.asarray(data, dtype=float)
        median = np.median(array, axis=1, keepdims=True)
        mad = np.median(np.abs(array - median), axis=1, keepdims=True)
        robust_std = 1.4826 * np.where(mad == 0, 1.0, mad)
        lower = median - threshold * robust_std
        upper = median + threshold * robust_std
        return np.clip(array, lower, upper)
