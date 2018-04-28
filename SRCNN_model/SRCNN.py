import numpy as np 
from keras.models import Model
from keras.layers import Input
from keras.layers.convolutional import Convolution2D
from keras import backend as K 
import keras.optimizers as optimizers
import data_prep as DP 

HR_data, LR_data = DP.get_data()

K.set_image_dim_ordering('th')
weights_path = ''

def SRCNN(n1=64,n2=32,f1=9,f2=1,f3=5,load_weights=False):
	inputs = Input(shape=(1,33,33))
	x = Convolution2D(n1, (f1,f1), activation='relu',padding='valid',name='level1')(inputs)
	x = Convolution2D(n2, (f2,f2), activation='relu',padding='valid',name='level2')(x)
	out = Convolution2D(1,(f3,f3), padding='valid', name='output')(x)

	model = Model(inputs,out)
	adam = optimizers.Adam(lr=1e-3)
	model.compile(optimizer=adam, loss='mse')#, metrics=[PSRNLoss])
	if load_weights:
		model.load_weights(weights_path)

	return model

def modelTrain(batch_size=128, nb_epochs=10):
	model = SRCNN()
	model.fit(x=LR_data,y=HR_data,batch_size=batch_size,epochs=nb_epochs)

	model_json = model.to_json()
	with open("model.json", "w") as json_file:
		json_file.write(model_json)

	model.save_weights("model.h5")
	print("Saved model to disk")

modelTrain()
print("Training over!!...Now test the model :P")


