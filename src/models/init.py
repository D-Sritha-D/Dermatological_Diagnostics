"""
Model architectures module for skin lesion classification
------------------------------------------------------
Includes CNN architecture, transfer learning, and ensemble components.
"""

from .cnn_architecture import DilatedCNN
from .transfer_learning import TransferLearningModel
from .ensemble import EnsembleModel, EnsembleTrainer

__all__ = [
    'DilatedCNN',
    'TransferLearningModel',
    'EnsembleModel',
    'EnsembleTrainer'
]

# Model configurations
MODEL_CONFIGS = {
    'dilated_cnn': {
        'filter_sizes': [64, 128, 256, 512, 512],
        'dilation_rates': [1, 2, 3],
        'dense_units': [1024, 512]
    },
    'transfer_learning': {
        'trainable_layers': 30,
        'dense_units': [1024, 512],
        'dropout_rate': 0.5
    }
}

def get_model_config(model_type):
    """Get configuration for specific model type"""
    return MODEL_CONFIGS.get(model_type, {})

def list_available_models():
    """List all available model architectures"""
    return list(MODEL_CONFIGS.keys())

def get_model_instance(model_type, **kwargs):
    """Factory function to create model instances"""
    if model_type == 'dilated_cnn':
        return DilatedCNN(**kwargs)
    elif model_type == 'transfer_learning':
        return TransferLearningModel(**kwargs)
    elif model_type == 'ensemble':
        return EnsembleModel(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
