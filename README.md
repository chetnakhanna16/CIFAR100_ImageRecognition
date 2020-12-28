# CIFAR100_ImageRecognition

# Version 2: CIFAR-100 Image Recognition Using Transfer Learning - EfficientNet-B0

## Abstract: 
Convolutional neural network (CNN) is a class of deep neural network commonly used to analyze images. The objective of this project is to build a convolutional neural network model that can correctly recognize and classify colored images of objects into one of the 100 available classes for CIFAR-100 dataset. The recognition of images in this project has been done using transfer learning approach. The network built in this project uses the state-of-the-art EfficientNet-B0 which was trained on the popular, challenging and large ImageNet dataset. Transfer learning and the idea of intelligently scaling the network (carefully balancing the network's width, depth and resolution) helped in getting a good performance on this dataset. By just training the model for 15 epochs, the model managed to achieve an accuracy of 82 percent. This is definitely a much better performance than the one achieved using a 9-layer convolutional neural network model trained for 100 epochs. The training of the model has been done on a GPU and the model has also been tested on some new random images to visualize the top 5 category predictions along with their probabilities.

## Glimpses of the task done and results achieved:

<B>Accuracy versus number of epochs</B>
![Accuracy versus number of epochs](https://github.com/chetnakhanna16/CIFAR100_ImageRecognition/blob/master/images/Accuracy_New.png)


<B>Loss versus number of epochs</B>
![Loss versus number of epochs](https://github.com/chetnakhanna16/CIFAR100_ImageRecognition/blob/master/images/Loss_New.png)


<B>True and Predicted labels</B>
![True and Predicted Labels](https://github.com/chetnakhanna16/CIFAR100_ImageRecognition/blob/master/images/TruePredictedLabels_1.png)


<B>Correct Prediction by the model</B>
![Correct Prediction by the Model](https://github.com/chetnakhanna16/CIFAR100_ImageRecognition/blob/master/images/Correct_Prediction2.png)


<B>Incorrect Prediction by the model</B>
![Incorrect Prediction by the Model](https://github.com/chetnakhanna16/CIFAR100_ImageRecognition/blob/master/images/Incorrect_Prediction2.png)


## CONCLUSION:
Recognition of different images is a simple task for we humans as it is easy for us to distinguish between different features. Somehow our brains are trained un-
consciously with a similar type of images that has helped us distinguish between features (images) without putting much effort into the task. For instance, after seeing a few cats, we can recognize almost every different type of cat we encounter in our life. However, machines need a lot of training for feature extraction which becomes a challenge due to high computation cost, memory requirement and processing power. 
The transfer learning model built in this project for CIFAR-100 dataset recognizes and classifies colored images of objects in one of the 100 available categories with 82% accuracy. The reported loss is 0.19. The model used techniques like early stopping, reduce learning rate on plateau and dropout to avoid overffitting. Even after training the model with millions of parameters, the model predicted the class for a few images completely wrong. It is considered that the performance of a deep learning model increases with the amount of data used in its training. It is believed that the accuracy of this dataset can be further improved by using the other versions of EfficientNet.

<B>Detailed Project Paper:</B>
https://github.com/chetnakhanna16/CIFAR100_ImageRecognition/blob/master/Cifar100_EfficientNetB0_ProjectPaper.pdf



# Version 1: CIFAR-100 Image Recognition Using Convolutional Neural Network

<B>Data Source:</B> https://www.cs.toronto.edu/~kriz/cifar.html

## Abstract: 
Convolutional neural network (CNN) is a class of deep neural network commonly used to analyze images. The objective of this project is to build a convolutional neural network model that can correctly recognize and classify colored images of objects into one of the 100 available classes for CIFAR-100 dataset. The recognition of images has been done using a 9-layer convolutional neural network model. The model uses techniques and processes like max pooling, zero padding, ReLU activation function (for hidden layers), Softmax activation function (for output layer) and Adam as the optimizer. In order to avoid overfitting, techniques and processes like dropout, early stopping have been used. As training a deep learning model on more data can lead to more skillful and robust model, so on-the-fly image data augmentation has been used to expand the dataset. By taking batches of size 64 and training the model for 100 epochs on a GPU, an accuracy of 59 percent has been achieved. The model has also been tested on some new random images to visualize the top 5 category predictions along with their probabilities.

## Glimpses of the task done and results achieved:

<B>Accuracy versus number of epochs</B>
![Accuracy versus number of epochs](https://github.com/chetnakhanna16/CIFAR100_ImageRecognition/blob/master/images/Accuracy_Final.png)

<B>Loss versus number of epochs</B>
![Loss versus number of epochs](https://github.com/chetnakhanna16/CIFAR100_ImageRecognition/blob/master/images/Loss_Final.png)

<B>True and Predicted labels</B>
![True and Predicted Labels](https://github.com/chetnakhanna16/CIFAR100_ImageRecognition/blob/master/images/TruePredictedLabels.png)

<B>Correct Prediction by the model</B>
![Correct Prediction by the Model](https://github.com/chetnakhanna16/CIFAR100_ImageRecognition/blob/master/images/Correct_Prediction.png)


<B>Incorrect Prediction by the model</B>
![Incorrect Prediction by the Model](https://github.com/chetnakhanna16/CIFAR100_ImageRecognition/blob/master/images/Incorrect_Prediction.png)



## CONCLUSION:
Recognition of different images is a simple task for we humans as it is easy for us to distinguish between different features. Somehow our brains are trained un-
consciously with a similar type of images that has helped us distinguish between features (images) without putting much effort into the task. For instance, after seeing a few cats, we can recognize almost every different type of cat we encounter in our life. However, machines need a lot of training for feature extraction which becomes a challenge due to high computation cost, memory requirement and processing power. The 9-layer deep neural network model built in this project for CIFAR-100 dataset recognizes and classifies colored images of objects in one of the 100 available categories with 59% accuracy. The ConvNet architecture of the model has three stacks of CONV-RELU layers followed by a POOL layer and then two fully connected (FC) RELU layers followed by a fully connected output layer. The model uses 13,870,484 trainable parameters which has been trained for an hour an half on a GPU with 8vCPUs. The Adam optimizer with learning rate 0.0001 and categorical cross entropy loss has been used to used to support the training process which involved 100 epochs and 64 as the batch size. The reported loss is 1.47. The model used techniques like early stopping and dropout to avoid overfitting. 
Even after training the model with millions of parameters, the model predicted the class for a few images completely wrong. As it is considered that the performance of a deep learning model increases with the amount of data used in its training, it would be highly possible that such a mediocre accuracy was due to the limited size of the dataset for each class. It is believed that the accuracy of this dataset can be further improved by working on different factors related to model building and hyperparameter tuning.

<B>Detailed Project Paper:</B>
https://github.com/chetnakhanna16/CIFAR100_ImageRecognition/blob/master/Project_Paper_001081074.pdf

