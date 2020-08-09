# CIFAR100_ImageRecognition
CIFAR-100 Image Recognition Using Convolutional Neural Network

Abstract: 

Convolutional neural network (CNN) is a class of deep neural network commonly used to analyze images. The objective of this project is to build a convolutional neural network model that can correctly recognize and classify colored images of objects into one of the 100 available classes for CIFAR-100 dataset. The recognition of images has been done using a 9-layer convolutional neural network model. The model uses techniques and processes like max pooling, zero padding, ReLU activation function (for hidden layers), Softmax activation function (for output layer) and Adam as the optimizer. In order to avoid overfitting, techniques and processes like dropout, early stopping have been used. As training a deep learning model on more data can lead to more skillful and robust model, so on-the-fly image data augmentation has been used to expand the dataset. By taking batches of size 64 and training the model for 100 epochs on a GPU, an accuracy of 59 percent has been achieved. The model has also been tested on some new random images to visualize the top 5 category predictions along with their probabilities.

Detailed Research Paper:
https://github.com/chetnakhanna16/CIFAR100_ImageRecognition/blob/master/Project_Paper_001081074.pdf
