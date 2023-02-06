import tensorflow as tf
from keras.models import Sequential
from keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D,Input,Dropout
from keras import backend as K

class Model1:
    @staticmethod
    def build(width,height,depth,classes):
        shape=(height,width,depth)
        baseModel = tf.keras.applications.vgg16.VGG16(weights="imagenet", include_top=False, input_tensor=Input(shape=(shape)))
        headModel = baseModel.output
        headModel = Flatten(name="flatten")(headModel)
        headModel = Dense(500, activation="relu")(headModel)
        headModel = Dropout(0.5)(headModel)
        headModel = Dense(classes, activation="softmax")(headModel)
        model = tf.keras.Model(inputs=baseModel.input, outputs=headModel)
        for layer in baseModel.layers:
	        layer.trainable = False
        
        return model    



