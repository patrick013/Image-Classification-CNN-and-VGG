import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import tensorflow as tf
from src.setting import *

if __name__== "__main__":

    path = input("Where is your image path? \n")
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMAGE_WIDTH, IMAGE_HEIGHT])
    image /= 255.0
    image_x=tf.expand_dims(image,axis=0)

    model=tf.keras.models.load_model('models/vggmodel.h5')

    y=model.predict(image_x)
    index=0
    if y >0.5:
        index=1
    else:
        index=0
    print("This image shows a "+ IDX_TO_LABELS[index])
