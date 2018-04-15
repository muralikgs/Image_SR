import tensorflow as tf 
import numpy as np 
import extract_imgs as EI 

print('Loading Data')
LR_data, HR_data = EI.get_data()

batch_size = 100

x = tf.placeholder('float')
y = tf.placeholder('float')

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME', data_format='NHWC')

def SR_model(x):
    weights = {'W_conv1':tf.Variable(tf.random_normal([9,9,1,64],stddev=1e-3),dtype=tf.float32),
               'W_conv2':tf.Variable(tf.random_normal([1,1,64,32],stddev=1e-3),dtype=tf.float32),
               'W_conv3':tf.Variable(tf.random_normal([5,5,32,1],stddev=1e-3),dtype=tf.float32)}


    biases = {'b_conv1': tf.Variable(tf.random_normal([64])), 'b_conv2': tf.Variable(tf.random_normal([32])), 'b_conv3': tf.Variable(tf.random_normal([1]))}
    conv1 = tf.nn.relu(conv2d(x, weights['W_conv1']) + biases['b_conv1'])
    conv2 = tf.nn.relu(conv2d(conv1, weights['W_conv2']) + biases['b_conv2'])
    output = conv2d(conv2, weights['W_conv3']) + biases['b_conv3']

    return output

def train_model(x):
	output = SR_model(x)
	saver = tf.train.Saver()
	cost = tf.reduce_mean(tf.squared_difference(output,y))
	optimizer = tf.train.RMSPropOptimizer(learning_rate=0.002).minimize(cost)	
	hm_epochs = 75
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		itern = int(int(LR_data.shape[0])/batch_size)
		for epoch in range(hm_epochs):
			epoch_loss = 0
			for i in range(itern):
				epoch_x, epoch_y = LR_data[i*batch_size:(i+1)*batch_size], HR_data[i*batch_size:(i+1)*batch_size]
				_, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
				epoch_loss += c

			print('Epoch', epoch+1, 'completed out of',hm_epochs,'loss:',epoch_loss)

		save_path = saver.save(sess,"./model_SR")


train_model(x)
print('Training Over')
