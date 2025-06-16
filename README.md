# Image_Classification_Model

*COMPANY*: CODTECH IT SOLUTIONS

*NAME*: MADDINENI ROHITHA

*INTERN ID*: CT06DL736

*DOMAIN*: MACHINE LEARNING

*DURATION*: 6 WEEEKS

*MENTOR*: NEELA SANTOSH

## Project Objective

This project demonstrates how to build a Convolutional Neural Network (CNN) using TensorFlow and Keras to perform image classification on the well-known CIFAR-10 dataset. CIFAR-10 is a collection of 60,000 32x32 color images in 10 classes, with 6,000 images per class. The model aims to learn from the training images and accurately predict the class of unseen test images.

##  Tools & Technologies Used:

- Python
- TensorFlow & Keras – for building and training the CNN
- Matplotlib & NumPy – for visualization and array processing
- CIFAR-10 Dataset – for image classification tasks

## Problem Statement:

The task is to build a neural network that can accurately classify small images into one of the following 10 categories:

- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

Each image in the CIFAR-10 dataset is 32x32 pixels with three color channels (RGB).

## Model Architecture:

The Convolutional Neural Network consists of the following layers:

1. Conv2D Layer (32 filters) with ReLU activation and kernel size 3x3
2. MaxPooling2D Layer with pool size 2x2
3. Conv2D Layer (64 filters) with ReLU activation
4. MaxPooling2D Layer
5. Conv2D Layer (64 filters) for deeper feature extraction
6. Flatten Layer to convert 2D feature maps into a 1D feature vector
7. Dense Layer (64 units) with ReLU activation
8. Dense Output Layer (10 units) with softmax activation for multi-class classification

The model uses the Adam optimizer, sparse categorical cross-entropy as the loss function (since the labels are integers), and accuracy as the evaluation metric.

## Data Preprocessing:

- The CIFAR-10 dataset is loaded using tensorflow.keras.datasets.
- Pixel values of the images are normalized to the range [0, 1] by dividing by 255.0.
- No manual reshaping is needed because the images are already in (32, 32, 3) shape.

## Training and Evaluation:

- The model is trained for 10 epochs using the model.fit() method.
- The training is performed on 50,000 images, and validation is done using 10,000 test images.
- After training, the model is evaluated on the test set using model.evaluate().

## Performance Visualization:

- A graph is plotted showing the training accuracy vs. validation accuracy over the 10 epochs using Matplotlib.
- This helps visualize how well the model is generalizing to unseen data and whether it's underfitting or overfitting.

## Final Output:

The model prints the final test accuracy, which shows how well it performs on unseen data. It also provides a graphical visualization of how accuracy changes over training epochs.

## How to Run:

1. Make sure you have Python 3.10 and TensorFlow installed.
2. Run the code in a Jupyter Notebook or any Python IDE.
3.The output will show training progress, final test accuracy, and a graph.

## Conclusion:

This project shows how a Convolutional Neural Network (CNN) can be used to classify images from the CIFAR-10 dataset. The model was built using TensorFlow and trained to recognize 10 different categories of objects, such as airplanes, cars, and animals.

After training, the model performed well on test data and achieved good accuracy. A graph of training and validation accuracy helped understand how the model improved over time.

This project is a good starting point for learning image classification using deep learning and can be improved further by adding more advanced techniques like data augmentation or deeper networks.
