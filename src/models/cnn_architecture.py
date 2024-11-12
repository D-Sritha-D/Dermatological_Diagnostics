import tensorflow as tf
from tensorflow.keras import layers, models

class DilatedCNN:
    """
    Implements the CNN architecture described in the paper
    with dilated convolutional units
    """
    def __init__(self, input_shape=(224, 224, 3), num_classes=9):
        self.input_shape = input_shape
        self.num_classes = num_classes
        
    def create_cbr_block(self, x, filters, dilation_rate):
        """Create CBr=iconv block as described in paper"""
        conv = layers.Conv2D(
            filters=filters,
            kernel_size=3,
            padding='same',
            dilation_rate=dilation_rate
        )(x)
        batch_norm = layers.BatchNormalization()(conv)
        relu = layers.ReLU()(batch_norm)
        return relu
    
    def build_model(self):
        """Build the complete model architecture"""
        # Input layer
        inputs = layers.Input(shape=self.input_shape)
        
        # Initial convolution
        x = self.create_cbr_block(inputs, filters=64, dilation_rate=1)
        
        # Create the five CBr=iconv blocks
        filter_sizes = [64, 128, 256, 512, 512]
        
        for filters in filter_sizes:
            # Create parallel dilated convolutions
            dilated_outputs = []
            for dilation_rate in [1, 2, 3]:
                dilated = self.create_cbr_block(x, filters, dilation_rate)
                dilated_outputs.append(dilated)
            
            # Concatenate dilated convolutions
            x = layers.Concatenate()(dilated_outputs)
            
            # Add max pooling
            x = layers.MaxPooling2D(pool_size=(2, 2), strides=2)(x)
        
        # Flatten and dense layers
        x = layers.Flatten()(x)
        x = layers.Dense(1024, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        
        # Output layer
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        # Create model
        model = models.Model(inputs=inputs, outputs=outputs)
        return model