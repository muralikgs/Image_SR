import numpy as np 
import cv2
import glob
import scipy.misc as misc

# Defining the parameters for preprocessing
f_sub = 33
scale = 2.0
stride = 14
input_size = 33
label_size = 21
pad = int((input_size - label_size) / 2)

def get_data():
	sub_img = list()
	sub_img_label = list()

	for img in glob.glob('../../Datasets/General100/*.png'):
		input_img = cv2.imread(img)

		# Converting to YCbCr space
		image = cv2.cvtColor(input_img, cv2.COLOR_BGR2YCR_CB)[:,:,0]

		# Generating the subimages
		h,w = image.shape
		h -= h%2
		w -= w%2
		image = image[0:h,0:w]

		scaled = misc.imresize(image, 1.0/scale, 'bicubic')
		scaled = misc.imresize(scaled, scale/1.0, 'bicubic')

		for i in range(0, h - input_size + 1, stride):
			for j in range(0, w - input_size + 1, stride):
				sub_img += [scaled[i:i+input_size,j:j+input_size]]
				sub_img_label += [image[i+pad:i+pad+label_size,j+pad:j+pad+label_size]]

	sub_img = np.array(sub_img)
	n1,n2,n3 = sub_img.shape
	LR_final = np.reshape(sub_img,(n1,1,n2,n3))

	sub_img_label = np.array(sub_img_label)
	n1,n2,n3 = sub_img_label.shape
	HR_final = np.reshape(sub_img_label,(n1,1,n2,n3))

	return HR_final/255, LR_final/255

def get_test_data():
	sub_img = list()
	sub_img_label = list()

	input_img = cv2.imread('../../Datasets/General100/im_008.png')

	# Converting to YCbCr space
	image = cv2.cvtColor(input_img, cv2.COLOR_BGR2YCR_CB)[:,:,0]

	# Generating the subimages
	h,w = image.shape
	h -= h%3
	w -= w%3
	image = image[0:h,0:w]

	scaled = misc.imresize(image, 1.0/scale, 'bicubic')
	scaled = misc.imresize(scaled, scale/1.0, 'bicubic')

	I = int((h - input_size) / stride) + 1
	J = int((w - input_size) / stride) + 1

	for i in range(0, h - input_size + 1, stride):
		for j in range(0, w - input_size + 1, stride):
			sub_img += [scaled[i:i+input_size,j:j+input_size]]
			sub_img_label += [image[i+pad:i+pad+label_size,j+pad:j+pad+label_size]]

	sub_img = np.array(sub_img)
	n1,n2,n3 = sub_img.shape
	LR_final = np.reshape(sub_img,(n1,1,n2,n3))

	sub_img_label = np.array(sub_img_label)
	n1,n2,n3 = sub_img_label.shape
	HR_final = np.reshape(sub_img_label,(n1,1,n2,n3))

	return HR_final/255, LR_final/255, I, J, h, w  


if __name__ == '__main__':
	HR_final, LR_final = get_data()
	print(HR_final.shape, LR_final.shape)
	cv2.imshow('HR', HR_final[75,0,:,:])
	cv2.imshow('LR', LR_final[75,0,:,:])

	cv2.waitKey(0)
	cv2.destroyAllWindows()
