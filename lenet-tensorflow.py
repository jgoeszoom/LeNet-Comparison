from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import matplotlib.pyplot as plt
import argparse as arg
import numpy as np
import sys
import timeit
import csv

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

Flag = None
Address = "mnist/model.ckpt"
epochs = 10000
batchsize = 50
img = list()
verdict = list()

def MyCNN(x):
	my_image = tf.reshape(x,[-1,28,28,1])

	Wconv1 = weight_variable([5,5,1,32])
	Bconv1 = bias_variable([32])
	conv1 = tf.nn.relu(conv2d(my_image, Wconv1)+ Bconv1)

	pool1 = max_pool_2x2(conv1)

	Wconv2 = weight_variable([5,5,32,64])
	Bconv2 = bias_variable([64])
	conv2 = tf.nn.relu(conv2d(pool1, Wconv2) + Bconv2)

	pool2 = max_pool_2x2(conv2)

	fcw1 = weight_variable([7*7*64, 1024])
	fcb1 = bias_variable([1024])
	
	pool2flattened = tf.reshape(pool2, [-1,7*7*64])
	fc1 = tf.nn.relu(tf.matmul(pool2flattened, fcw1) + fcb1)

	probkeep = tf.placeholder(tf.float32)
	dropout = tf.nn.dropout(fc1, probkeep)

	fcw2 = weight_variable([1024,10])
	fcb2 = bias_variable([10])

	y_solution = tf.matmul(dropout, fcw2) + fcb2
	return y_solution, probkeep

def conv2d(x,W):
	return tf.nn.conv2d(x, W, strides = [1,1,1,1], padding = 'SAME')

def max_pool_2x2(x,):
	return tf.nn.max_pool(x, ksize = [1,2,2,1], strides =  [1,2,2,1], padding = 'SAME')

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev = 0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape = shape)
	return tf.Variable(initial)

def main(_):
	mnist = input_data.read_data_sets(Flags.data_dir, one_hot = True)
	tf.reset_default_graph()

	x = tf.placeholder(tf.float32, [None, 784])
	y = tf.placeholder(tf.float32, [None, 10])

	y_solution, keep_prob = MyCNN(x)

	cross_entropy = tf.reduce_mean(
		tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = y_solution))
	train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
	correct_prediction = tf.equal(tf.argmax(y_solution,1),tf.argmax(y,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	saver = tf.train.Saver()

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		try:
			saver.restore(sess, Address)
			print("Model found in", Address)
		except:
			print("Model not found. Moving on to create the first model!")
		'''
		for i in range(epochs):
			batch = mnist.train.next_batch(batchsize)
			if i % 100 == 0:
				train_accuracy = accuracy.eval(feed_dict = {
					x: batch[0], y: batch[1], keep_prob:1.0})
				print('step %d, training accuracy %g' % (i, train_accuracy))
			train_step.run(feed_dict = {x: batch[0], y: batch[1], keep_prob: 0.5})
		
		print('test accuracy %g' % accuracy.eval(feed_dict = {x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0}))
		'''
		
		prediction = tf.argmax(y_solution,1)
		
		t = timeit.Timer()
		alpha = prediction.eval(feed_dict={x:mnist.test.images, y: mnist.test.labels, keep_prob: 1.0})
		print(t.timeit(2))
		i = input("Please enter an index to check (please use an integer I can't prevent string inputs with my bad coding skills holy my life is falling apart at least I have Kermit): ")	
		print("MNIST image #: ")
		print(i)
		print("Predicted value: ")
		print(alpha[i])
		print("Labeled value: ")
		Nothot = mnist.test.labels[i]
		
		nou = 0
		Noot = False
		while(Noot == False):
			if (Nothot[nou] == 1):
				Noot = True
				break
			else:
				nou += 1	
		print(nou)	
					
		tmp = saver.save(sess,Address)
		print("Model saved to", tmp)

if __name__ == '__main__':
	#start = time.clock()
	parser = arg.ArgumentParser()
	parser.add_argument('--data_dir', type = str, default = 'input_ data', help = 'Directory for storing input_data')
	Flags, unparsed = parser.parse_known_args()
	main(parser)
	#print (time.clock()) 















	
















