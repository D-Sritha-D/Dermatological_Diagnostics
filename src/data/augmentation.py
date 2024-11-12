import albumentations as A
import numpy as np
import cv2

class DataAugmentation:
    """
    Implements data augmentation pipeline following paper specifications
    """
    def __init__(self):
        self.augmentation = A.Compose([
            # Geometric augmentations
            A.RandomRotate90(p=0.5),
            A.Flip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.2,
                rotate_limit=45,
                p=0.5
            ),
            
            # Color augmentations
            A.OneOf([
                A.RandomBrightness(limit=0.2),
                A.RandomContrast(limit=0.2),
                A.HueSaturationValue(
                    hue_shift_limit=20,
                    sat_shift_limit=30,
                    val_shift_limit=20
                )
            ], p=0.5),
            
            # Noise augmentations
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0)),
                A.ISONoise(),
                A.MultiplicativeNoise()
            ], p=0.5),
            
            # Elastic transformations
            A.ElasticTransform(
                alpha=120,
                sigma=120 * 0.05,
                alpha_affine=120 * 0.03,
                p=0.3
            )
        ])
    
    def augment(self, image):
        """Apply augmentation to image"""
        augmented = self.augmentation(image=image)
        return augmented['image']
    
