# COVID-19 Detection Using Convolutional Neural Networks (CNN)

## Overview
This project involves the development and training of a CNN model to classify images as COVID-19 positive, pneumonia, or normal. The model is built using PyTorch and fine-tunes a pretrained ResNet-50 architecture to achieve accurate predictions on the provided dataset.

## Project Structure
- **Dataset Source**: [Kaggle COVID-19 Image Dataset](https://www.kaggle.com/datasets/pranavraikokte/covid19-image-dataset)
- **Model Architecture**: ResNet-50 (pretrained)
- **Development Environment**: Google Colab with GPU support
- **Programming Language**: Python (with PyTorch)

## Key Features
- **Data Preprocessing**:
  - Images resized to 224x224.
  - Normalization based on ImageNet statistics.
  - Data augmentation includes random horizontal flips for training data.

- **Model Training**:
  - Frozen base layers of ResNet-50.
  - Fine-tuned the final fully connected layer to output predictions for three classes.
  - Training and validation loops implemented with cross-entropy loss and Adam optimizer.

- **Evaluation**:
  - Model evaluated over 10 epochs with metrics logged for training loss, training accuracy, validation loss, and validation accuracy.

- **Inference**:
  - Model can make predictions on unseen data, displaying the predicted label and confidence score.

## Setup Instructions
1. Clone the repository or download the project files.
2. Ensure you have the following dependencies installed:
   ```bash
   pip install torch torchvision matplotlib kaggle
```
