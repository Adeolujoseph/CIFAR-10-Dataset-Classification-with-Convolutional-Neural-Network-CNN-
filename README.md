# CIFAR-10-Dataset-Classification-with-Convolutional-Neural-Network-CNN
classifying the cifar10 dataset using cnn

## Overview
This project involves training a Convolutional Neural Network (CNN) on the CIFAR-10 dataset for image classification. The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 different classes, with 6,000 images per class.

## Dataset Description
Dataset Name: CIFAR-10
Number of Classes: 10
Image Size: 32x32 pixels
Channels: RGB (3 channels)
Classes:
Airplane
Automobile
Bird
Cat
Deer
Dog
Frog
Horse
Ship
Truck.

The dataset is directly downloaded using the TorchVision library's datasets.CIFAR10 module.

## Dependencies
Python 3.x
PyTorch
Torch
Torchvision
Matplotlib
NumPy

## Preprocessing

Random affine transformations including rotation, translation, and scaling

Random resized crop to a size of 32x32 pixels, with a scaling factor ranging from 0.75 to 1 times the original size.

## Model Architecture
This project implements a Convolutional Neural Network (CNN) architecture designed for image classification tasks, specifically trained on the CIFAR-10 dataset. The neural network architecture, defined in the Net class, comprises a series of convolutional layers, pooling operations, batch normalization, dropout, fully connected layers, and activation functions.

### Neural Network Layers and Operations

Convolutional Layers:

Conv1: Applies 32 filters of size 3x3 to the input RGB images (3 input channels), resulting in 32 output channels.

Conv2: Utilizes 64 filters of size 3x3 on the previous layer's output, generating 64 output channels.

Conv3: Employs 128 filters of size 3x3, producing 128 output channels.

Conv4: Similar to Conv3, utilizing 128 filters.

Conv5: Applies 256 filters of size 3x3, generating 256 output channels.

Conv6: Similar to Conv5, utilizing 256 filters.

### Pooling Layers:

Max pooling operations with a kernel size of 2x2 are applied after specific convolutional layers (pool1, pool2, pool3).

### Normalization and Regularization:

Batch normalization layers (bn1, bn2, ..., bn6) stabilize and accelerate training after convolutional layers.

Dropout of 20% (Dropout2d(p=0.2)) is used after the third pooling layer to prevent overfitting.

### Fully Connected Layers (FC):
Flattened feature maps after the last pooling layer are passed through a sequence of fully connected layers:
fc → fc2 → fc3 → fc4 → fc5 → fc6 → fc7.
The final fully connected layer (fc7) has num_classes output nodes, suitable for multi-class classification tasks.

### Activation Function:
Rectified Linear Unit (ReLU) activation function is applied after each convolutional layer and fully connected layer (except for the final layer fc7). This introduces non-linearity to the model.

## Training and Evaluation

Loss Criteria: CrossEntropyLoss function 

Optimizer: Adam optimizer 

Number of Epochs: 30

Evaluation Metric: Accuracy


## Result 
86.83% was achieved after 30 epochs. Max val accuracy during training was 87.18% @ EPOCH 27. 

Training loss and test loss was visualized as epoch number progressed to observe overfitting. No significant changes in validation loss after epoch 10

## Project Limitations and Future Work
I believe the accuracy can get better by further hyperparameter tuning or using pre-trained models




