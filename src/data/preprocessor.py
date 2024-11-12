import cv2
import numpy as np

class DataProcessor:
    """
    Handles image preprocessing according to paper specifications
    """
    def __init__(self, image_size=(224, 224)):
        self.image_size = image_size
    
    def apply_poisson_process(self, image):
        """Apply Poisson process to reduce noise"""
        # Convert to float32
        image_float = image.astype(np.float32) / 255.0
        # Add Poisson noise and denoise
        noisy = np.random.poisson(image_float * 255) / 255
        denoised = cv2.fastNlMeansDenoisingColored(noisy.astype(np.float32), None, 10, 10, 7, 21)
        return denoised
    
    def apply_decorrelation(self, image):
        """Apply decorrelation filter"""
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        # Normalize L channel
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        # Merge channels
        enhanced = cv2.merge([l, a, b])
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
    
    def apply_median_filter(self, image):
        """Apply median filter"""
        return cv2.medianBlur((image * 255).astype(np.uint8), 3)
    
    def load_and_preprocess(self, image_path):
        """Load and preprocess a single image"""
        # Read image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize
        image = cv2.resize(image, self.image_size)
        
        # Apply enhancement pipeline
        image = self.apply_poisson_process(image)
        image = self.apply_decorrelation(image)
        image = self.apply_median_filter(image)
        
        # Normalize
        image = (image - np.mean(image)) / np.std(image)
        
        return image