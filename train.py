import os
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from src.data.data_generator import DataGenerator
from src.models.transfer_learning import TransferLearningModel
from src.models.ensemble import EnsembleTrainer
from src.utils.evaluation import ModelEvaluator
from src.utils.visualization import ResultVisualizer
from config import *

def get_image_paths_and_labels(data_dir):
    """Get image paths and labels from directory structure"""
    image_paths = []
    labels = []
    
    for class_idx, class_name in enumerate(os.listdir(data_dir)):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
            
        for image_name in os.listdir(class_dir):
            if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(class_dir, image_name))
                labels.append(class_idx)
    
    return np.array(image_paths), np.array(labels)

def main():
    # Set random seeds for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)
    
    # Get image paths and labels
    print("Loading dataset...")
    image_paths, labels = get_image_paths_and_labels(DATA_DIR)
    
    # Split dataset
    train_paths, test_paths, train_labels, test_labels = train_test_split(
        image_paths, labels, test_size=TEST_SPLIT, stratify=labels, random_state=42
    )
    
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_paths, train_labels, test_size=VALIDATION_SPLIT, 
        stratify=train_labels, random_state=42
    )
    
    # Create data generators
    train_generator = DataGenerator(
        train_paths, train_labels,
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        augment=True
    )
    
    val_generator = DataGenerator(
        val_paths, val_labels,
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        augment=False
    )
    
    test_generator = DataGenerator(
        test_paths, test_labels,
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        augment=False
    )
    
    # Create and train models
    print("\nInitializing transfer learning models...")
    transfer_learning = TransferLearningModel(
        input_shape=(*IMAGE_SIZE, 3),
        num_classes=NUM_CLASSES
    )
    ensemble_models = transfer_learning.create_ensemble()
    
    print("\nTraining ensemble...")
    trainer = EnsembleTrainer(ensemble_models, batch_size=BATCH_SIZE)
    histories = trainer.train_ensemble(train_generator, val_generator)
    
    # Save models
    print("\nSaving models...")
    for i, model in enumerate(ensemble_models):
        model.save(os.path.join(SAVED_MODELS_DIR, f'model_{i+1}.h5'))
    
    # Evaluate ensemble
    print("\nEvaluating ensemble...")
    evaluator = ModelEvaluator(ensemble_models[0], CLASS_NAMES)
    visualizer = ResultVisualizer(RESULTS_DIR)
    
    # Calculate metrics
    conf_matrix, class_metrics, y_pred_probs, y_true = evaluator.calculate_metrics(test_generator)
    
    # Save metrics
    evaluator.save_metrics(class_metrics, RESULTS_DIR)
    
    # Create visualizations
    visualizer.plot_confusion_matrix(conf_matrix, CLASS_NAMES)
    visualizer.plot_roc_curves(y_true, y_pred_probs, CLASS_NAMES)
    
    for i, history in enumerate(histories):
        visualizer.plot_training_history(history)
    
    print("\nTraining complete! Results saved in:", RESULTS_DIR)

if __name__ == "__main__":
    main()