import numpy as np
import tensorflow as tf

class EnsembleModel:
    """
    Handles ensemble predictions and training
    """
    def __init__(self, models, weights=None):
        self.models = models
        self.weights = weights if weights is not None else [1/len(models)] * len(models)
    
    def predict(self, X):
        """Get ensemble predictions"""
        predictions = []
        
        for model, weight in zip(self.models, self.weights):
            pred = model.predict(X)
            predictions.append(pred * weight)
        
        # Average predictions
        ensemble_pred = np.sum(predictions, axis=0)
        return ensemble_pred

class EnsembleTrainer:
    """
    Handles training of the ensemble model
    """
    def __init__(self, models, batch_size=32):
        self.models = models
        self.batch_size = batch_size
    
    def train_model(self, model, train_generator, val_generator, epochs=100):
        """Train individual model"""
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.1,
                patience=5
            )
        ]
        
        history = model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=epochs,
            callbacks=callbacks
        )
        
        return history
    
    def train_ensemble(self, train_generator, val_generator):
        """Train all models in ensemble"""
        histories = []
        
        for i, model in enumerate(self.models):
            print(f"\nTraining model {i+1}/{len(self.models)}")
            history = self.train_model(
                model, train_generator, val_generator
            )
            histories.append(history)
        
        return histories