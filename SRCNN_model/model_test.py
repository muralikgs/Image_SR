import numpy as np 
from keras.models import Model
from keras.layers import Input
from keras.layers.convolutional import Convolution2D
from keras import backend as K 
from keras.models import model_from_json 
import keras.optimizers as optimizers
import data_prep as DP 
import cv2
import h5py
import os

stride = 14
input_size = 33
label_size = 21
pad = int((input_size - label_size)/2)

HR_data, LR_data, I, J, n1, n2 = DP.get_test_data()

_,_,l1,l2 = LR_data.shape

K.set_image_dim_ordering('th')
weights_path = 'model.h5'

def outpatchCombine(patches):
	mat = np.zeros((n1,n2))
	req_img = np.zeros((n1,n2))

	s1 = int((n1 - input_size + 1) / stride)
	s2 = int((n2 - input_size + 1) / stride)

	for i in range(0, n1 - input_size + 1, stride):
		for j in range(0, n2 - input_size + 1, stride):
			mat[i+pad:i+pad+label_size,j+pad:j+pad+label_size] += 1

	mat = mat[pad:s1*stride + input_size - pad, pad:s2*stride + input_size - pad]
	mat_inv = (1/mat)

	for i in range(0, n1 - input_size + 1, stride):
		for j in range(0, n2 - input_size + 1, stride):
			req_img[i+pad:i+pad+label_size,j+pad:j+pad+label_size] += patches[int(i/stride)*J + int(j/stride),0,:,:]

	req_img = req_img[pad:s1*stride + input_size - pad, pad:s2*stride + input_size - pad]
	req_img = req_img * mat_inv

	return req_img

def lrpatchCombine(patches):
	mat = np.zeros((n1,n2))
	req_img = np.zeros((n1,n2))

	s1 = int((n1 - input_size + 1) / stride)
	s2 = int((n2 - input_size + 1) / stride)

	for i in range(0, n1 - input_size + 1, stride):
		for j in range(0, n2 - input_size + 1, stride):
			mat[i:i+input_size,j:j+input_size] += 1

	mat_inv = (1/mat)

	for i in range(0, n1 - input_size + 1, stride):
		for j in range(0, n2 - input_size + 1, stride):
			req_img[i:i+input_size,j:j+input_size] += patches[int(i/stride)*J + int(j/stride),0,:,:]

	req_img = req_img * mat_inv

	return req_img

def SRCNN(n1=64,n2=32,f1=9,f2=1,f3=5,load_weights=False):
	inputs = Input(shape=(1,l1,l2))
	x = Convolution2D(n1, (f1,f1), activation='relu',padding='valid',name='level1')(inputs)
	x = Convolution2D(n2, (f2,f2), activation='relu',padding='valid',name='level2')(x)
	out = Convolution2D(1,(f3,f3), padding='valid', name='output')(x)

	model = Model(inputs,out)
	adam = optimizers.Adam(lr=1e-3)
	model.compile(optimizer=adam, loss='mse')#, metrics=[PSRNLoss])
	if load_weights:
		weights = []
		f = h5py.File(os.path.expanduser(weights_path), mode='r')

		for layer_name in f.attrs['layer_names']:
			g = f[layer_name]
			for weight_name in g.attrs['weight_names']:
				weights.append(g[weight_name].value)

		#weights[0] = weights[0].transpose((2,3,1,0))
		#weights[2] = weights[2].transpose((2,3,1,0))
		#weights[4] = weights[4].transpose((2,3,1,0))
		
		model.set_weights(weights)
		f.close()

	return model

def modelTest():
	model = SRCNN(load_weights=True)
	print("Weights loaded from the disk")
	output = model.predict(LR_data,verbose=0)
	
	return output

if __name__ == '__main__':
	output = np.abs(modelTest())
	print(output.shape)
	
	img = (outpatchCombine(output)*255).astype(np.uint8)
	LR = (lrpatchCombine(LR_data)*255).astype(np.uint8)
	HR = (outpatchCombine(HR_data)*255).astype(np.uint8)

	cv2.imshow('HR', HR)
	cv2.imshow('LR', LR)
	cv2.imshow('output', img)

	cv2.waitKey(0)

	cv2.destroyAllWindows()
