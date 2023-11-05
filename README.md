# Moroccan Money Classification with PyTorch

This repository contains code for a Convolutional Neural Network (CNN) model in PyTorch to classify Moroccan money. The code loads a dataset from Kaggle, prepares the data, builds a CNN model, trains the model, and evaluates its performance.

## Getting Started

### Prerequisites

Make sure you have the required libraries installed:

- PyTorch
- torchvision
- matplotlib
- numpy

You can install these libraries using pip:

```bash
pip install torch torchvision matplotlib numpy
```
## Dataset

The dataset used in this project is [Moroccan Money dataset](https://www.kaggle.com/datasets/oussamaouardini/moroccan-money-dataset). It contains images of Moroccan money that have been resized and normalized for training and testing.

## Training the Model

To train the CNN model, run the provided code in the `Moroccan_Money_classification.ipynb` Jupyter Notebook. The code includes data loading, model creation, training loop, and saving the trained model.

## Model Architecture

The CNN model architecture used for this project:

- Convolutional Layer 1: 3 input channels, 6 output channels, kernel size 5x5
- Max Pooling Layer: 2x2
- Convolutional Layer 2: 6 input channels, 16 output channels, kernel size 5x5
- Fully Connected Layer 1: Input size 2704, Output size 120
- Fully Connected Layer 2: Input size 120, Output size 84
- Fully Connected Layer 3: Input size 84, Output size 11 (number of classes)

## Training and Evaluation

The model is trained using stochastic gradient descent (SGD) with a cross-entropy loss function. The training process is saved in a history dictionary, containing loss and accuracy values. The trained model is saved to a file named `Moroccan_Money_detection.pth`.

The model's performance is evaluated on a test dataset, and accuracy metrics are provided for each class.

## Visualization

The `Moroccan_Money_classification.ipynb` includes visualization of training loss and accuracy using matplotlib.

## Acknowledgments

This project is for educational purposes and uses a Kaggle dataset. Credit to the Kaggle community for providing the dataset.

Enjoy experimenting with Moroccan money classification using PyTorch!

## Author

Mohamed Stifi
[Linkedin](https://www.linkedin.com/in/mohamed-stifi-3636b0258/)
