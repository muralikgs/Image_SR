import cv2
import numpy as np 
import scipy.misc
import matplotlib.pyplot as plt 

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def unpack_data(file):
	batch = unpickle(file)
	x = batch[b'data']
	print('Reading data from ', file)
	output_images = np.zeros((10000,32,32))
	input_images = np.zeros((10000,32,32))
	for i in range(x.shape[0]):
		img = x[i,:]
		temp = np.transpose(np.reshape(img,(3, 32,32)), (1,2,0))
		output_images[i,:,:] = cv2.cvtColor(temp, cv2.COLOR_RGB2YCR_CB)[:,:,0]
		inter_image = scipy.misc.imresize(output_images[i],0.5, interp = 'nearest')
		input_images[i,:,:] = scipy.misc.imresize(inter_image,2.0, interp = 'bicubic') 	

	input_images = np.reshape(input_images, (10000,32,32,1))
	output_images = np.reshape(output_images, (10000,32,32,1))
	return input_images/255, output_images/255

	#return input_images, output_images 

def get_data():
	lr1, hr1 = unpack_data('./Data_Cifar/data_batch_1') 
	lr2, hr2 = unpack_data('./Data_Cifar/data_batch_2') 
	#lr3, hr3 = unpack_data('./Data_Cifar/data_batch_3') 
	#lr4, hr4 = unpack_data('./Data_Cifar/data_batch_4') 
	#lr5, hr5 = unpack_data('./Data_Cifar/data_batch_5')

	LR_data = np.concatenate([lr1,lr2], axis = 0) 
	HR_data = np.concatenate([hr1,hr2], axis = 0)

	return LR_data, HR_data


if __name__ == '__main__':
	LR_data, HR_data = get_data()

	print(LR_data.shape, HR_data.shape)
	LR_data = LR_data.astype(np.uint8)
	HR_data = HR_data.astype(np.uint8)

	# Displaying the 100th image
	'''
	plt.imshow(LR_data[100,:,:,:])
	plt.axis('off')
	plt.figure()
	plt.imshow(HR_data[100,:,:,:])
	plt.axis('off')

	plt.show()
	'''
	cv2.imshow('HR',HR_data[100,:,:,0])
	cv2.imshow('LR',LR_data[100,:,:,0])
	k = cv2.waitKey(0)

	cv2.destroyAllWindows()

