
from visutil import PlotTraining
from tensorflow.keras.applications import VGG16
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
import os,random
print(tf.__version__)

IMAGE_WIDTH=112
IMAGE_HEIGHT=112
BATCH_SIZE=64
EPOCHS=20


class Dataset_Initializer():

    def _get_pathframe(self, path):
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
            if category == 'dog':
                categories.append(1)
            else:
                categories.append(0)

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
        image = tf.image.resize(image, [192, 192])
        image /= 255.0  # normalize to [0,1] range
        return image

    def _convert_to_tensor(self, df):
        '''
        Convert each data and labels to tensor
        '''
        path_ds = tf.data.Dataset.from_tensor_slices(df['paths'])
        image_ds = path_ds.map(self._load_and_preprocess_image)
        label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(df['category'], tf.int64))
        return image_ds, label_ds

    def init_dataset(self, path):
        images_paths_df = self._get_pathframe(path)

        return self._convert_to_tensor(images_paths_df)

class Models():
    def __init__(self, model='sim_vgg'):
        self._model=model

    def _vggmodel(self):
        pre_trained_model = VGG16(input_shape=(124, 124, 3), include_top=False, weights="imagenet")

        for layer in pre_trained_model.layers[:15]:
            layer.trainable = False

        for layer in pre_trained_model.layers[15:]:
            layer.trainable = True

        last_layer = pre_trained_model.get_layer('block5_pool')
        last_output = last_layer.output
        x = layers.Flatten()(last_output)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(1, activation='sigmoid')(x)

        vggmodel = tf.keras.models.Model(pre_trained_model.input, x)

        vggmodel.compile(loss='binary_crossentropy',
                         optimizer=tf.keras.optimizers.SGD(lr=1e-4, momentum=0.9),
                         metrics=['accuracy'])

        vggmodel.summary()

        return model

    def _mymodel(self):

        model = tf.keras.models.Sequential()
        model.add(
            layers.Conv2D(8, (3, 3), padding='same', activation='relu', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 3)))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Conv2D(16, (3, 3), padding='same', activation='relu'))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))

        opt = tf.keras.optimizers.Adam(0.001)
        model.compile(optimizer=opt,
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        return model

    def get_model(self):
        dic={'vgg':self._vggmodel(),
            'cnn':self._mymodel()}
        model=dic[self._model]

        return model

'''
Train the model
'''
dInit=Dataset_Initializer()
datapath ='dataset/train/'
X,Y=dInit.init_dataset(datapath)
dataset=tf.data.Dataset.zip((X,Y)).shuffle(1000).batch(BATCH_SIZE, drop_remainder=True)
dataset_train=dataset.take(22500)
dataset_valid=dataset.skip(22500)

model=Models()
hist=model.fit_generator(dataset_train,epochs=EPOCHS,validation_data=dataset_valid)
# model.save('DogCat_ImageClassification/model')