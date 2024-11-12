# Dermatological Diagnostics: A Unified Deep Learning Framework for Skin Lesion and Cancer Classification

A comprehensive implementation of the research paper on skin lesion classification using Convolutional Neural Networks (CNNs) and transfer learning. This project provides a robust solution for classifying nine different types of skin lesions.

## Features

- Multi-model ensemble approach using MobileNet, InceptionV3, and InceptionResNetV2
- Advanced data augmentation techniques
- Custom CNN architecture with dilated convolutions
- Comprehensive evaluation metrics and visualizations
- Support for both script-based and notebook-based execution

## Skin Lesion Classes

The system can classify the following nine types of skin lesions:
1. Squamous cell carcinoma
2. Dermatofibroma
3. Actinic Keratosis
4. Pigmented benign keratosis
5. Nevus
6. Vascular Lesion
7. Seborrheic Keratosis
8. Basal Cell Carcinoma
9. Melanoma

## Project Structure

```
skin_lesion_classifier/
│
├── data/
│   └── skin_lesion_dataset/
│       ├── squamous_cell_carcinoma/
│       ├── dermatofibroma/
│       ├── actinic_keratosis/
│       ├── pigmented_benign_keratosis/
│       ├── nevus/
│       ├── vascular_lesion/
│       ├── seborrheic_keratosis/
│       ├── basal_cell_carcinoma/
│       └── melanoma/
│
├── src/
│   ├── data/
│   │   ├── augmentation.py
│   │   ├── data_generator.py
│   │   └── preprocessor.py
│   │
│   ├── models/
│   │   ├── cnn_architecture.py
│   │   ├── transfer_learning.py
│   │   └── ensemble.py
│   │
│   └── utils/
│       ├── evaluation.py
│       └── visualization.py
│
├── notebooks/
│   └── model_development.ipynb
│
├── saved_models/
├── results/
├── requirements.txt
├── config.py
├── train.py
└── predict.py
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/skin-lesion-classifier.git
cd skin-lesion-classifier
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # For Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create required directories:
```bash
mkdir -p skin_lesion_classifier/{data/skin_lesion_dataset,src/{data,models,utils},notebooks,saved_models,results}
```

## Dataset Setup

1. Organize your dataset in the following structure:
```
data/skin_lesion_dataset/
├── squamous_cell_carcinoma/    # 181 images
├── dermatofibroma/             # 95 images
├── actinic_keratosis/          # 114 images
├── pigmented_benign_keratosis/ # 462 images
├── nevus/                      # 357 images
├── vascular_lesion/            # 139 images
├── seborrheic_keratosis/       # 77 images
├── basal_cell_carcinoma/       # 376 images
└── melanoma/                   # 438 images
```

2. Verify setup by running:
```bash
python test_setup.py
```

## Usage

### Option 1: Using Python Scripts

1. Train the model:
```bash
python train.py
```

2. Make predictions:
```bash
python predict.py
```

### Option 2: Using Jupyter Notebook

1. Start Jupyter:
```bash
jupyter notebook
```

2. Navigate to `notebooks/model_development.ipynb`

3. Run cells sequentially for:
- Data analysis
- Model training
- Evaluation
- Visualization

## Model Architecture

The system uses an ensemble of models:
1. Custom CNN with dilated convolutions
2. Transfer learning models:
   - MobileNet
   - InceptionV3
   - InceptionResNetV2

## Evaluation Metrics

The model evaluates:
- Accuracy
- Sensitivity
- Specificity
- AUC-ROC
- Confusion Matrix

## Results Visualization

Results are saved in the `results/` directory:
- Confusion matrices
- ROC curves
- Training history plots
- Per-class metrics

## Common Issues & Solutions

1. Memory Issues
   - Reduce batch size in `config.py`
   - Use data generator instead of loading all images

2. CUDA Out of Memory
   - Reduce image size
   - Decrease batch size
   - Use fewer models in ensemble

3. Dataset Issues
   - Ensure correct folder structure
   - Verify image formats (.jpg, .png, .jpeg)
   - Check class distribution

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this code in your research, please ensure you cite it.
