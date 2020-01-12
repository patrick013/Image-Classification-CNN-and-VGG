# Image Classification - Cat or Dog

Image classification refers to a process in computer vision that can classify an image according to its visual content. For example, an image classification algorithm may be designed to tell if an image contains a human figure or not. While detecting an object is trivial for humans, robust image classification is still a challenge in computer vision applications. This is a image classification project based on [Cat of Dog Kaggel competition](https://www.kaggle.com/c/dogs-vs-cats/overview/description).
## Prediction
```python
python predict.py
>>> Where is your image path?
>>> samples/dog.jpg
>>> This image shows a dog
```
## Details 
For details please check [ImageClassification.ipynb](https://github.com/patrick013/Image-Classification-CNN-and-VGG/blob/master/Dog_vs_Cat.ipynb).
In this project, I applied convolutional neural network for image classification. And the end, I also applied VGG-16 convolutional neural network by using transfer learning. VGG-16 is a popular deep learning neural network in computer vision, proposed by K. Simonyan and A. Zisserman from the University of Oxford in the paper “[Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)”. The structrue of VGG 16 network is following:
![VGG](https://neurohive.io/wp-content/uploads/2018/11/vgg16-1-e1542731207177.png)

## Result 
Result of VGG16 using transfer learning on 20 epochs.
![result](https://raw.githubusercontent.com/patrick013/Image-Classification-CNN-and-VGG/master/pictures/a.png)

## Big Challenge - Loss does not change

The biggest challenge that I met was that there was not a sign that the loss tended to decrease at all after 10 epochs when I was using VGG network. I spent whole night to fix this problem and tried many suggestions online, such as check the dataset labels, reducing the networks and so on. However, it didn't work at all. One of the reason was that I forgot to convert y label to one hot array.

## Reference

>Simonyan, Karen, and Andrew Zisserman. "Very deep convolutional networks for large-scale image recognition." arXiv preprint arXiv:1409.1556 (2014).
