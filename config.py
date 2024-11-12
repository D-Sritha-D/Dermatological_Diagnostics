import os

# Path configurations
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'skin_lesion_dataset')
SAVED_MODELS_DIR = os.path.join(BASE_DIR, 'saved_models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

# Model configurations
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 100
NUM_CLASSES = 9

# Class names
CLASS_NAMES = [
    'Squamous cell carcinoma',
    'Dermatofibroma',
    'Actinic Keratosis',
    'Pigmented benign keratosis',
    'Nevus',
    'Vascular Lesion',
    'Seborrheic Keratosis',
    'Basal Cell Carcinoma',
    'Melanoma'
]

# Model parameters
TRANSFER_LEARNING_MODELS = [
    'mobilenet',
    'inception_v3',
    'inception_resnet_v2'
]

# Training parameters
LEARNING_RATE = 1e-4
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.2

# Create required directories
for directory in [DATA_DIR, SAVED_MODELS_DIR, RESULTS_DIR]:
    os.makedirs(directory, exist_ok=True)