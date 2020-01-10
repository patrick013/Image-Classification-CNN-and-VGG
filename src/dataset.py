import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from .setting import *

class LoadByDataframe():
    '''
    Load dataset from one directory by using Pandas
    '''
    def __init__(self,
                 path=DIRECTORY_DATASET):

        self._df=self._get_pathframe(path)

    def _get_pathframe(self,path):
        '''
        Get all the images paths and its corresponding labels
        Store them in pandas dataframe
        '''
        filenames = os.listdir(path)
        categories = []
        paths = []
        for filename in filenames:
            paths.append(path + filename)
            category = filename.split('.')[0]
            if category == 'dog': categories.append(1)
            else: categories.append(0)

        df = pd.DataFrame({
            'filename': filenames,
            'category': categories,
            'paths': paths
        })
        return df

    def _load_and_preprocess_image(self, path):
        '''
        Load each image and resize it to desired shape
        '''
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, [IMAGE_WIDTH, IMAGE_HEIGHT])
        image /= 255.0  # normalize to [0,1] range
        return image

    def load_dataset(self):
        '''
        Convert each data and labels to tensor
        '''
        path_ds = tf.data.Dataset.from_tensor_slices(self._df['paths'])
        image_ds = path_ds.map(self._load_and_preprocess_image)
        # onehot_label=tf.one_hot(tf.cast(df['category'], tf.int64),2) if using softmax
        onehot_label = tf.cast(self._df['category'], tf.int64)
        label_ds = tf.data.Dataset.from_tensor_slices(onehot_label)

        dataset = tf.data.Dataset.zip((image_ds, label_ds)).shuffle(buffer_size=BUFFER_SIZE)
        dataset_train = dataset.take(NUM_TRAIN).batch(BATCH_SIZE, drop_remainder=True)
        dataset_validation = dataset.skip(NUM_TRAIN).batch(BATCH_SIZE, drop_remainder=True)

        return dataset_train, dataset_validation

class LoadByGenerator():
    '''
    Data Augmentation:
    One way to fix overfitting is to augment the dataset so that it has a sufficient number of training examples.
    Data augmentation takes the approach of generating more training data from existing training samples by
    augmenting the samples using random transformations that yield believable-looking images.
    The goal is the model will never see the exact same picture twice during training. This helps expose the model to more aspects of the data and generalize better.
    '''

    def __init__(self, path= DIRECTORY_DATASET_SEPERATED):
        self._image_gen_train = ImageDataGenerator(
            rescale=1. / 255,
            rotation_range=45, # rotation
            width_shift_range=.15,
            height_shift_range=.15,
            horizontal_flip=True, # apply horizontal_flip
            zoom_range=0.5 # apply zoom
        )
        self._path_training=path+'train'
        self._path_validation=path+'validation'

    def load_dataset(self):

        train_data_gen = self._image_gen_train.flow_from_directory(batch_size=BATCH_SIZE,
                                                                  directory=self._path_training,
                                                                  shuffle=True,
                                                                  target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
                                                                  class_mode='binary')
        validation_data_gen = self._image_gen_train.flow_from_directory(batch_size=BATCH_SIZE,
                                                                  directory=self._path_validation,
                                                                  shuffle=True,
                                                                  target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
                                                                  class_mode='binary')
        return train_data_gen,validation_data_gen


if __name__== "__main__":
    loader=LoadByDataframe('../dataset/dataset_seperated')
    training_set, validation_set= loader.load_dataset()
    print(training_set)