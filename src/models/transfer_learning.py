from tensorflow.keras import applications, layers, models

class TransferLearningModel:
    """
    Implements transfer learning approach using multiple pre-trained models
    """
    def __init__(self, input_shape=(224, 224, 3), num_classes=9):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.base_models = {
            'mobilenet': applications.MobileNet,
            'inception_v3': applications.InceptionV3,
            'inception_resnet_v2': applications.InceptionResNetV2
        }
    
    def create_model(self, model_name, trainable_layers=30):
        """Create transfer learning model with specified base"""
        # Get base model
        base_model = self.base_models[model_name](
            include_top=False,
            weights='imagenet',
            input_shape=self.input_shape
        )
        
        # Freeze early layers
        for layer in base_model.layers[:-trainable_layers]:
            layer.trainable = False
        
        # Create complete model
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(1024, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        return model
    
    def create_ensemble(self):
        """Create ensemble of all transfer learning models"""
        models_list = []
        
        for model_name in self.base_models.keys():
            model = self.create_model(model_name)
            models_list.append(model)
        
        return models_list