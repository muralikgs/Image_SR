import tensorflow as tf 
import numpy as np
import extract_imgs as EI 
import cv2

print('Loading Data')
LR_data, HR_data = EI.get_data()

x = tf.placeholder('float')

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME', data_format='NHWC')

def SR_model(x):
    weights = {'W_conv1':tf.Variable(tf.random_normal([9,9,1,64])),
               'W_conv2':tf.Variable(tf.random_normal([1,1,64,32])),
               'W_conv3':tf.Variable(tf.random_normal([5,5,32,1]))}


    biases = {'b_conv1': tf.Variable(tf.random_normal([64])), 'b_conv2': tf.Variable(tf.random_normal([32])), 'b_conv3': tf.Variable(tf.random_normal([1]))}
    conv1 = tf.nn.relu(conv2d(x, weights['W_conv1']) + biases['b_conv1'])
    conv2 = tf.nn.relu(conv2d(conv1, weights['W_conv2']) + biases['b_conv2'])
    output = conv2d(conv2, weights['W_conv3']) + biases['b_conv3']

#    return output
    return output, weights

def model_test(x):
	output, weights = SR_model(x)
	saver = tf.train.Saver()
	with tf.Session() as sess:
		#sess.run(tf.global_variables_initializer())
		saver.restore(sess,"./model_SR")
		img = np.reshape(sess.run(output, feed_dict={x: np.reshape(LR_data[100],(1,32,32,1))}),(32,32))
		img = (img*255).astype(np.uint8)
		#print('Output', img)
		#print(HR_data[100,:,:,0])

		error = np.sum((img - HR_data[100,:,:,0])**2)
		print(error)
		cv2.imshow('HR',(HR_data[100,:,:,0]*255).astype(np.uint8))
		cv2.imshow('Output', img)
		
		cv2.waitKey(0)

model_test(x)
print('Test Complete')
cv2.destroyAllWindows()