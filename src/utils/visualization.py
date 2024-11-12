import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, auc
import os
from datetime import datetime

class ResultVisualizer:
    """
    Handles visualization of results and metrics
    """
    def __init__(self, save_dir):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    def plot_confusion_matrix(self, conf_matrix, class_names):
        """Plot and save confusion matrix"""
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            conf_matrix,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names
        )
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(os.path.join(self.save_dir, f'confusion_matrix_{timestamp}.png'))
        plt.close()
    
    def plot_roc_curves(self, y_true, y_pred_probs, class_names):
        """Plot and save ROC curves"""
        plt.figure(figsize=(10, 8))
        
        for i, class_name in enumerate(class_names):
            # Convert to one-hot if needed
            if len(y_true.shape) == 1:
                y_true_binary = (y_true == i).astype(int)
            else:
                y_true_binary = y_true[:, i]
            
            fpr, tpr, _ = roc_curve(y_true_binary, y_pred_probs[:, i])
            roc_auc = auc(fpr, tpr)
            
            plt.plot(
                fpr, tpr,
                label=f'{class_name} (AUC = {roc_auc:.2f})'
            )
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend(loc="lower right")
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(os.path.join(self.save_dir, f'roc_curves_{timestamp}.png'))
        plt.close()
    
    def plot_training_history(self, history):
        """Plot and save training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot accuracy
        ax1.plot(history.history['accuracy'], label='Training')
        ax1.plot(history.history['val_accuracy'], label='Validation')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        
        # Plot loss
        ax2.plot(history.history['loss'], label='Training')
        ax2.plot(history.history['val_loss'], label='Validation')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(os.path.join(self.save_dir, f'training_history_{timestamp}.png'))
        plt.close()