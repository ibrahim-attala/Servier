import tensorflow as tf
from keras.models import Sequential
from keras.layers import BatchNormalization, Conv2D, MaxPooling2D, Activation, Flatten, Dropout, Dense
from keras import backend as K

class Model2:
	@staticmethod
	def build(width,height,depth,classes):
		model=Sequential()
		shape=(height,width,depth)
		channelDim=-1

		if K.image_data_format()=="channels_first":
			shape=(depth,height,width)
			channelDim=1

		model.add(Conv2D(32, (3,3), padding="same",input_shape=shape))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=channelDim))
		model.add(MaxPooling2D(pool_size=(2,2)))
		model.add(Dropout(0.25))

		model.add(Conv2D(64, (3,3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=channelDim))
		model.add(Conv2D(64, (3,3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=channelDim))
		model.add(MaxPooling2D(pool_size=(2,2)))
		model.add(Dropout(0.25))

		model.add(Conv2D(128, (3,3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=channelDim))
		model.add(Conv2D(128, (3,3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=channelDim))
		model.add(Conv2D(128, (3,3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=channelDim))
		model.add(MaxPooling2D(pool_size=(2,2)))
		model.add(Dropout(0.25))

		model.add(Flatten())
		model.add(Dense(500))
		model.add(Activation("relu"))
		model.add(BatchNormalization())
		model.add(Dropout(0.5))

		model.add(Dense(classes))
		model.add(Activation("softmax"))

		return model
