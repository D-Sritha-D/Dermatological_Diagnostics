import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

class ModelEvaluator:
    """
    Handles model evaluation metrics and visualization
    """
    def __init__(self, model, class_names):
        self.model = model
        self.class_names = class_names
        
    def calculate_metrics(self, test_generator):
        """Calculate comprehensive metrics"""
        # Get predictions
        y_pred_probs = self.model.predict(test_generator)
        y_pred = np.argmax(y_pred_probs, axis=1)
        y_true = np.argmax(np.concatenate([y for _, y in test_generator], axis=0), axis=1)
        
        # Calculate metrics
        conf_matrix = confusion_matrix(y_true, y_pred)
        
        # Per-class metrics
        class_metrics = {}
        for i, class_name in enumerate(self.class_names):
            # Calculate true positives, false positives, etc.
            tp = conf_matrix[i, i]
            fp = conf_matrix[:, i].sum() - tp
            fn = conf_matrix[i, :].sum() - tp
            tn = conf_matrix.sum() - (tp + fp + fn)
            
            # Calculate metrics
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
            
            class_metrics[class_name] = {
                'Sensitivity': sensitivity,
                'Specificity': specificity,
                'Accuracy': accuracy,
                'Precision': precision,
                'F1-Score': f1_score
            }
        
        return conf_matrix, class_metrics, y_pred_probs, y_true
    
    def save_metrics(self, class_metrics, save_dir):
        """Save metrics to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics_file = os.path.join(save_dir, f'metrics_{timestamp}.txt')
        
        with open(metrics_file, 'w') as f:
            f.write("Model Evaluation Metrics\n")
            f.write("=" * 50 + "\n\n")
            
            for class_name, metrics in class_metrics.items():
                f.write(f"\n{class_name}:\n")
                f.write("-" * 30 + "\n")
                for metric_name, value in metrics.items():
                    f.write(f"{metric_name}: {value:.4f}\n")