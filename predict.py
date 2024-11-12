import tensorflow as tf
import numpy as np
import cv2
import os
from src.data.preprocessor import DataProcessor
from src.models.ensemble import EnsembleModel
from config import *

class LesionPredictor:
    """
    Handles prediction for new images
    """
    def __init__(self, model_dir):
        self.preprocessor = DataProcessor(IMAGE_SIZE)
        self.models = self.load_models(model_dir)
        self.ensemble = EnsembleModel(self.models)
    
    def load_models(self, model_dir):
        """Load all saved models"""
        models = []
        for model_file in sorted(os.listdir(model_dir)):
            if model_file.endswith('.h5'):
                model_path = os.path.join(model_dir, model_file)
                model = tf.keras.models.load_model(model_path)
                models.append(model)
        return models
    
    def predict_image(self, image_path):
        """Predict class for a single image"""
        # Preprocess image
        image = self.preprocessor.load_and_preprocess(image_path)
        image = np.expand_dims(image, axis=0)
        
        # Get prediction
        pred_probs = self.ensemble.predict(image)
        pred_class = np.argmax(pred_probs)
        confidence = pred_probs[0][pred_class]
        
        return CLASS_NAMES[pred_class], confidence, pred_probs[0]

def main():
    # Initialize predictor
    predictor = LesionPredictor(SAVED_MODELS_DIR)
    
    while True:
        # Get image path from user
        image_path = input("\nEnter image path (or 'q' to quit): ")
        
        if image_path.lower() == 'q':
            break
        
        if not os.path.exists(image_path):
            print("Error: File does not exist")
            continue
        
        try:
            # Get prediction
            predicted_class, confidence, probabilities = predictor.predict_image(image_path)
            
            # Print results
            print("\nPrediction Results:")
            print("-" * 50)
            print(f"Predicted Class: {predicted_class}")
            print(f"Confidence: {confidence:.2%}")
            print("\nClass Probabilities:")
            for class_name, prob in zip(CLASS_NAMES, probabilities):
                print(f"{class_name}: {prob:.2%}")
            
        except Exception as e:
            print(f"Error processing image: {str(e)}")

if __name__ == "__main__":
    main()