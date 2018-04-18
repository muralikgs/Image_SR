import glob
import cv2
import numpy as np 
import scipy.misc

f_sub = 34

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

def get_data():
	img_final = list()
	Lowres = list()
# Loading all the images from the folder ../Datasets/General100

# In the following line replace the path with the path to the folder
# containing the General100 dataset
	for img in glob.glob('../../Datasets/General100/*.png'): 
		input_img = cv2.imread(img)

# Converting to YCbCr space
		img_Y = cv2.cvtColor(input_img, cv2.COLOR_BGR2YCR_CB)[:,:,0]

# Generating the subimages of size 33 X 33 and stacking them
		img_new = imgCrop(img_Y)
		inter_image = scipy.misc.imresize(img_new,0.5,interp='nearest')
		LR_image = scipy.misc.imresize(inter_image,2.0,interp='bicubic')
		
		for i in range(int(img_new.shape[0]/f_sub)):
			for j in range(int(img_new.shape[1]/f_sub)):
				img_final += [img_new[i*f_sub:(i+1)*f_sub,j*f_sub:(j+1)*f_sub]]
				Lowres += [LR_image[i*f_sub:(i+1)*f_sub,j*f_sub:(j+1)*f_sub]]
			
	img_final = np.array(img_final)
	n1,n2,n3 = img_final.shape
	HR_final = np.reshape(img_final,(n1,n2,n3,1))

	Lowres = np.array(Lowres)
	LR_final = np.reshape(Lowres,(n1,n2,n3,1))

	return HR_final/255, LR_final/255 

if __name__ == '__main__':
	HR_final, LR_final = get_data()
	print(HR_final.shape, LR_final.shape)
	cv2.imshow('HR', HR_final[75,:,:,0])
	cv2.imshow('LR', LR_final[75,:,:,0])

	cv2.waitKey(0)
	cv2.destroyAllWindows()