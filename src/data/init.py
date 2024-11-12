"""
Data handling module for skin lesion classification
------------------------------------------------
Includes data preprocessing, augmentation, and generator components.
"""

from .augmentation import DataAugmentation
from .data_generator import DataGenerator
from .preprocessor import DataProcessor

__all__ = [
    'DataAugmentation',
    'DataGenerator',
    'DataProcessor'
]

# Default configurations
DEFAULT_IMAGE_SIZE = (224, 224)
DEFAULT_BATCH_SIZE = 32
DEFAULT_AUGMENTATION_PROBABILITY = 0.5

class DataConfig:
    """Configuration class for data-related parameters"""
    def __init__(self):
        self.image_size = DEFAULT_IMAGE_SIZE
        self.batch_size = DEFAULT_BATCH_SIZE
        self.augmentation_probability = DEFAULT_AUGMENTATION_PROBABILITY

# Global configuration instance
config = DataConfig()

def set_image_size(size):
    """Set global image size"""
    config.image_size = size

def set_batch_size(size):
    """Set global batch size"""
    config.batch_size = size

def get_config():
    """Get current data configuration"""
    return config