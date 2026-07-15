"""Dataset splitting utilities."""

import numpy as np
from typing import Tuple, List


class DatasetSplitter:
    """Split dataset into train/val/test sets."""
    
    def __init__(self, train_size: float = 0.7, val_size: float = 0.15, test_size: float = 0.15):
        """Initialize splitter.
        
        Args:
            train_size: Training set proportion
            val_size: Validation set proportion
            test_size: Test set proportion
        """
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
    
    def split(self, data: List, labels: List = None) -> Tuple:
        """Split data into train/val/test.
        
        Args:
            data: Input data
            labels: Optional labels
            
        Returns:
            Tuple of (train, val, test) indices
        """
        pass