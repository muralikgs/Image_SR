import numpy as np 
import cv2 
import data_prep as DP 

HR_data,_, I, J, n1, n2 = DP.get_test_data()
print(n1,n2)

stride = 14
input_size = 33
label_size = 21
pad = int((input_size - label_size)/2)

mat = np.zeros((n1,n2))
s1 = int((n1 - input_size + 1) / stride)
s2 = int((n2 - input_size + 1) / stride)

print (s1,s2,I,J)
for i in range(0,n1 - input_size + 1, stride):
	for j in range(0, n2 - input_size + 1, stride):
		mat[i+pad:i+pad+label_size,j+pad:j+pad+label_size] += 1

mat = mat[pad:s1*stride + input_size - pad, pad:s2*stride + input_size - pad]
img_mat = 1/mat

HR_image = np.zeros((n1,n2))

for i in range(0,n1 - input_size + 1, stride):
	for j in range(0, n2 - input_size + 1, stride):
		HR_image[i+pad:i+pad+label_size,j+pad:j+pad+label_size] += HR_data[int(i/stride)*J + int(j/stride),0,:,:]
		
HR_image = HR_image[pad:s1*stride + input_size - pad, pad:s2*stride + input_size - pad]
HR_image = HR_image * img_mat

img_mat = ((1/mat)*255).astype(np.uint8)

img = (HR_image*255).astype(np.uint8)
cv2.imshow('img',img)
cv2.imshow('Matrix', img_mat)
cv2.waitKey(0)

cv2.destroyAllWindows()

