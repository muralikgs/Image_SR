import glob
import cv2
import numpy as np 

f_sub = 33

def imgCrop(img):
	nr, nc = img.shape
	
	rr = nr % f_sub
	rc = nc % f_sub

	rc_l = int(rc/2)
	rc_r = rc - rc_l

	rr_l = int(rr/2)
	rr_r = rr - rr_l

	img_new = img[rr_l:nr-rr_r,rc_l:nc-rc_r]

	return img_new

def cvtSubImg():
	img_final = list()

# Loading all the images from the folder ../Datasets/General100

# In the following line replace the path with the path to the folder
# containing the General100 dataset
	for img in glob.glob('../../Datasets/General100/*.png'): 
		input_img = cv2.imread(img)

# Converting to YCbCr space
		img_Y = cv2.cvtColor(input_img, cv2.COLOR_BGR2YCR_CB)[:,:,0]

# Generating the subimages of size 33 X 33 and stacking them
		img_new = imgCrop(img_Y)
	
		for i in range(int(img_new.shape[0]/f_sub)):
			for j in range(int(img_new.shape[1]/f_sub)):
				img_final += [img_new[i*f_sub:(i+1)*f_sub,j*f_sub:(j+1)*f_sub]]


	img_final = np.array(img_final)
	return img_final

if __name__ == '__main__':
	train_data = cvtSubImg()
	print(train_data.shape)
	cv2.imshow('Image', train_data[75])

cv2.waitKey(0)

cv2.destroyAllWindows()