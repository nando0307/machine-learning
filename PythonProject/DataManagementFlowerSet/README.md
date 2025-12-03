# Flower Classification CNN

A comprehensive deep learning project for classifying 102 species of flowers using Convolutional Neural Networks (CNN) with PyTorch.

## Project Overview

This project implements a flower classification system using:
- **Dataset**: Oxford 102 Flower Dataset
- **Framework**: PyTorch
- **Model**: Custom CNN architecture with regularization techniques
- **Features**: Data augmentation, train/val/test splits, comprehensive training pipeline

## Dataset

The project uses the **Oxford 102 Flower Dataset**, which contains 8,189 images across 102 flower species.

### Downloading the Dataset

The dataset is automatically downloaded when you run the notebook. However, if you need to download it manually, use these URLs:

**Image Archive:**
```
https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz
```

**Labels File:**
```
https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat
```

**To download manually:**

1. Download the image archive and extract it:
   ```bash
   cd DataManagementFlowerSet
   wget https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz
   tar -xzf 102flowers.tgz
   ```

2. Download the labels file:
   ```bash
   wget https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat
   ```

3. The extracted structure should look like:
   ```
   DataManagementFlowerSet/
   ├── flower_data/
   │   ├── jpg/
   │   │   └── image_*.jpg (8,189 images)
   │   ├── imagelabels.mat
   │   └── labels_description.txt
   ├── flower_classifier.ipynb
   ├── helper_utils.py
   └── README.md
   ```

### Automatic Download

The `download_dataset()` function in the notebook handles automatic downloading:
- Checks if dataset already exists locally
- Downloads only if files are missing
- Extracts images to `flower_data/jpg/` directory
- Downloads and saves labels to `flower_data/imagelabels.mat`

## Project Structure

### Files

- **flower_classifier.ipynb**: Main Jupyter notebook with complete training pipeline
- **helper_utils.py**: Utility functions for visualization and data processing
- **flower_data/**: Dataset directory (created during download)

### Key Components

#### Dataset Handling
- `FlowerDataset`: Custom PyTorch Dataset class for loading flower images
- `RobustFlowerDataset`: Error-handling variant with validation
- `MonitoredDataset`: Performance monitoring and statistics
- `SubsetWithTransform`: Applies different transforms to train/val/test splits

#### Data Processing
- `get_dataloaders()`: Creates train/val/test dataloaders with proper transforms
- `get_augmentation_transform()`: Implements data augmentation pipeline
- `Denormalize`: Inverse normalization for visualization

#### Model Architecture
- **FlowerClassifier**: CNN with 3 convolutional blocks and FC layers
  - Conv blocks: 3→32→64→128 channels
  - Dropout: 0.7 on FC layers for regularization
  - L2 weight decay: 0.0001 to 0.001

- **ImprovedFlowerClassifier**: Enhanced architecture with BatchNormalization
  - BatchNorm2d after each conv block
  - Dropout2d(0.2) on conv layers
  - Global average pooling
  - More parameter-efficient design

#### Training Functions
- `train_epoch()`: Single epoch training with progress tracking
- `evaluate()`: Validation/test evaluation with accuracy metrics
- `training_loop()`: Complete train/val loop with metrics collection

## Dependencies

```
torch==2.9.1
torchvision==0.24.1
torchaudio==2.9.1
numpy
matplotlib
scipy
Pillow
scikit-learn
requests
ipywidgets
```

## Training Configuration

### Hyperparameters
- **Batch Size**: 32
- **Learning Rate**: 0.001
- **Optimizer**: Adam with weight_decay=0.001
- **Loss Function**: CrossEntropyLoss
- **Epochs**: 15
- **Scheduler**: ReduceLROnPlateau (factor=0.5, patience=3)

### Data Splits
- **Training**: 70% (5,732 images) - with augmentation
- **Validation**: 15% (1,228 images) - without augmentation
- **Testing**: 15% (1,229 images) - without augmentation

### Regularization Techniques
1. **Dropout**: 0.7 on fully connected layers, 0.2 on convolutional layers
2. **BatchNormalization**: Applied after convolutional blocks
3. **L2 Regularization**: weight_decay=0.001 in optimizer
4. **Learning Rate Scheduling**: Reduces LR on validation loss plateau
5. **Data Augmentation**: RandomHorizontalFlip, RandomRotation, ColorJitter

## Data Augmentation Pipeline

Applied only to training set:
- **RandomHorizontalFlip**: 50% probability
- **RandomRotation**: ±10 degrees
- **ColorJitter**: Brightness adjustment (0.2)
- **Normalization**: ImageNet standard (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

## Quick Start

1. Clone the repository and navigate to the project:
   ```bash
   cd DataManagementFlowerSet
   ```

2. Install dependencies:
   ```bash
   pip install torch torchvision torchaudio numpy matplotlib scipy Pillow scikit-learn requests ipywidgets
   ```

3. Open the notebook:
   ```bash
   jupyter notebook flower_classifier.ipynb
   ```

4. Run all cells - the dataset will be downloaded automatically on first run

## Model Training

To train the model, use the `training_loop()` function:

```python
trained_model, metrics = training_loop(
    model=model,
    train_loader=train_dataloader,
    val_loader=val_dataloader,
    loss_function=loss_function,
    optimizer=optimizer,
    num_epochs=15,
    device=device
)

train_losses, val_losses, val_accuracies = metrics
```

## Evaluation

Evaluate on test set:

```python
test_accuracy = evaluate(trained_model, test_dataloader, device)
print(f"Test Accuracy: {test_accuracy:.2f}%")
```

## Features

✓ Complete data pipeline with automatic download
✓ Multiple dataset classes with error handling
✓ Comprehensive data augmentation
✓ Multiple CNN architectures with regularization
✓ Training with validation and test evaluation
✓ Performance monitoring and statistics
✓ Overfitting analysis and solutions
✓ Reusable dataloader creation function
✓ Detailed progress tracking

## Notes

- First run will download ~370MB of data (images + labels)
- Requires ~500MB disk space for extracted dataset
- Training on GPU (CUDA/MPS) is recommended for speed
- The notebook includes fallback to CPU if GPU unavailable
- Dataset automatically cached after first download

## License

The Oxford 102 Flower Dataset is provided by the University of Oxford.
