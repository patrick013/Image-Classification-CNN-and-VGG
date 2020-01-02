# Image Classification - Cat or Dog Kaggle Competition

![catanddog](https://miro.medium.com/max/2844/1*hCxU4nK6ulpnwhSpgWiPPg.png)

This is a image classification project from [Cat of Dog Kaggel competition](https://www.kaggle.com/c/dogs-vs-cats/overview/description). In this project, I applied convolutional neural network for image classification. And the end, I also applied VGG-16 convolutional neural network by using transfer learning. VGG-16 is a popular deep learning neural network in computer vision, proposed by K. Simonyan and A. Zisserman from the University of Oxford in the paper “[Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)”. The structrue of VGG 16 network is following:
![VGG](https://neurohive.io/wp-content/uploads/2018/11/vgg16-1-e1542731207177.png)

## Details 
For details please check [ImageClassification.ipynb](https://github.com/patrick013/Image-Classification-CNN-and-VGG/blob/master/ImageClassification.ipynb)

## Result 
Result of VGG16 using transfer learning on 20 epochs.
![result](https://raw.githubusercontent.com/patrick013/Image-Classification-CNN-and-VGG/master/pictures/a.png)

## Prediction
> python predict.py

## Summary
Obviously, VGG wins! My model to some degress is overfitted, which means the accuracy on training set is higher than that on validation set. To solve this problem:
1. Batch Normalization
2. Dropout
3. Regularization

### Big Challenge - Loss does not change

The biggest challenge that I met was that there was not a sign that the loss tended to decrease at all after 10 epochs when I was using VGG network. I spent whole night to fix this problem and tried many suggestions online, such as check the dataset labels, reducing the networks and so on. However, it didn't work at all. One of the reason was that I forgot to convert y label to one hot array.

## Reference

>Simonyan, Karen, and Andrew Zisserman. "Very deep convolutional networks for large-scale image recognition." arXiv preprint arXiv:1409.1556 (2014).
