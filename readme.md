# Facial Expression Recognition using CNN

## Overview
This project aims to build a Convolutional Neural Network (CNN) model for facial expression recognition. The model is trained on a dataset containing grayscale images of faces categorized into three classes: happy, sad, and neutral expressions.

## Requirements
- Python 3.x
- TensorFlow 2.x
- Keras
- Matplotlib
- NumPy
- Pandas
- Seaborn

## Installation
1. Clone the repository to your local machine:
2. Install the required packages:

## Dataset
The dataset is organized into two directories:
- `faces/dataset/train`: Contains training images.
- `faces/dataset/validation`: Contains validation images.

## Model Architecture
The CNN model architecture consists of multiple layers:
- Convolutional layers with batch normalization, ReLU activation, max pooling, and dropout.
- Fully connected layers with batch normalization, ReLU activation, and dropout.
- Output layer with softmax activation for multiclass classification.

## Training
1. Preprocess images using `ImageDataGenerator` to rescale pixel values.
2. Define the model architecture and compile it using the Adam optimizer.
3. Train the model using the training dataset (`train_ds`) for 30 epochs with early stopping based on validation loss.

## Evaluation
After training, the model's performance is evaluated using the validation dataset (`val_ds`). The training history is plotted to visualize the loss and accuracy over epochs.

## Model Saving
The trained model is saved in two formats:
- `model.h5`: Complete model with architecture and weights.
- `model_weight.h5`: Only the model weights.

## Files Included
- `facial_expression_recognition.ipynb`: Jupyter Notebook containing the entire code.
- `model.h5`: Trained model in HDF5 format.
- `model_weight.h5`: Model weights only.
- `README.md`: This README file.

## Authors
- 2335
- 2336
- 2337
- 2343
- 2345


