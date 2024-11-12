"""
Utility functions module for skin lesion classification
---------------------------------------------------
Includes evaluation metrics and visualization tools.
"""

from .evaluation import ModelEvaluator
from .visualization import ResultVisualizer

__all__ = [
    'ModelEvaluator',
    'ResultVisualizer'
]

# Metric configurations
METRIC_CONFIGS = {
    'classification': [
        'accuracy',
        'precision',
        'recall',
        'f1_score',
        'auc_roc'
    ],
    'visualization': [
        'confusion_matrix',
        'roc_curve',
        'precision_recall_curve',
        'training_history'
    ]
}

# Visualization configurations
VISUALIZATION_CONFIGS = {
    'figure_size': {
        'confusion_matrix': (12, 10),
        'roc_curve': (10, 8),
        'training_history': (15, 5)
    },
    'color_schemes': {
        'confusion_matrix': 'Blues',
        'roc_curve': 'tab10',
        'training_history': 'Set1'
    }
}

def get_metric_config(metric_type):
    """Get configuration for specific metric type"""
    return METRIC_CONFIGS.get(metric_type, [])

def get_visualization_config(vis_type):
    """Get configuration for specific visualization type"""
    return {
        'figure_size': VISUALIZATION_CONFIGS['figure_size'].get(vis_type),
        'color_scheme': VISUALIZATION_CONFIGS['color_schemes'].get(vis_type)
    }

# Utility functions
def setup_logging(log_dir=None):
    """Setup logging configuration"""
    import logging
    import os
    from datetime import datetime
    
    if log_dir is None:
        log_dir = 'logs'
    
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'run_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

# Error handling
class EvaluationError(Exception):
    """Exception raised for errors in evaluation process"""
    pass

class VisualizationError(Exception):
    """Exception raised for errors in visualization process"""
    pass
