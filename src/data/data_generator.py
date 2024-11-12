import tensorflow as tf
import numpy as np
from .augmentation import DataAugmentation
from .preprocessor import DataProcessor

class DataGenerator(tf.keras.utils.Sequence):
    """
    Custom data generator with augmentation and preprocessing
    """
    def __init__(self, 
                 image_paths, 
                 labels, 
                 batch_size=32, 
                 image_size=(224, 224), 
                 augment=True,
                 shuffle=True):
        self.image_paths = image_paths
        self.labels = labels
        self.batch_size = batch_size
        self.image_size = image_size
        self.augment = augment
        self.shuffle = shuffle
        self.augmenter = DataAugmentation()
        self.preprocessor = DataProcessor(image_size)
        self.on_epoch_end()
    
    def __len__(self):
        """Number of batches per epoch"""
        return int(np.ceil(len(self.image_paths) / self.batch_size))
    
    def __getitem__(self, idx):
        """Get batch at index idx"""
        # Get batch indexes
        start_idx = idx * self.batch_size
        end_idx = min((idx + 1) * self.batch_size, len(self.image_paths))
        batch_indexes = self.indexes[start_idx:end_idx]
        
        # Initialize batch arrays
        batch_x = np.empty((len(batch_indexes), *self.image_size, 3))
        batch_y = np.empty(len(batch_indexes))
        
        # Generate batch data
        for i, idx in enumerate(batch_indexes):
            # Load and preprocess image
            image = self.preprocessor.load_and_preprocess(self.image_paths[idx])
            
            # Apply augmentation if needed
            if self.augment:
                image = self.augmenter.augment(image)
            
            batch_x[i] = image
            batch_y[i] = self.labels[idx]
        
        return batch_x, tf.keras.utils.to_categorical(batch_y, num_classes=9)
    
    def on_epoch_end(self):
        """Called at the end of every epoch"""
        self.indexes = np.arange(len(self.image_paths))
        if self.shuffle:
            np.random.shuffle(self.indexes)