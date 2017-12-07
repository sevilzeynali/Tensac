#!/usr/bin/env python
# -*- coding: utf-8 -*-
# importing parameters

import tensorflow as tf
import numpy as np
from numpy import array
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' 
os.environ['CUDA_VISIBLE_DEVICES'] = ''#disable cuda
TRAIN_DATA_PATH=''
TRAIN_LABEL_PATH=''
TEST_DATA_PATH=''
TEST_LABEL_PATH=''
PREDICTION_PATH=''
LOG_PATH = ''
MODEL_PATH=''

#charging train set data
data_train=np.genfromtxt(TRAIN_DATA_PATH)

#Converting train data set labels to One-hot
file_of_labels=open(TRAIN_LABEL_PATH,'r')
liste_of_labels=[line.rstrip('\n') for line in file_of_labels]
liste_of_labels=array(liste_of_labels)
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(liste_of_labels)
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded),1)
onehot_data_label = onehot_encoder.fit_transform(integer_encoded)

#importer les datas de test et les labels de test
data_test=np.genfromtxt(TEST_DATA_PATH)

#Converting test data set labels to One-hot
file_of_labels_test=open(TEST_LABEL_PATH,'r')
fileOut=open(PREDICTION_PATH,'w')
liste_of_labels_test=[line.rstrip('\n') for line in file_of_labels_test]
liste_of_labels_test=array(liste_of_labels_test)
label_encoder = LabelEncoder()
integer_encoded_test = label_encoder.fit_transform(liste_of_labels_test)
onehot_encoder_test = OneHotEncoder(sparse=False)
integer_encoded_test = integer_encoded_test.reshape(len(integer_encoded_test),1)
onehot_test_labels = onehot_encoder_test.fit_transform(integer_encoded_test)

learning_rate = 0.001
training_epochs = 7
batch_size = 70
display_step = 1


# Network Parameters
n_hidden_1 = 200 # 1st layer 
n_hidden_2 = 200 # 2nd layer 
n_input = 300 # Number of feature
n_classes = 2 # Number of classes to predict

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])
# Create model
def multilayer_perceptron(x, weights, biases):
	# Hidden layer with RELU activation
	with tf.name_scope("activation_layers"):
		layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
		layer_1 = tf.nn.relu(layer_1)
		tf.summary.histogram("activation_layer1", layer_1)
	# Hidden layer with RELU activation
		layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
		layer_2 = tf.nn.relu(layer_2)
		tf.summary.histogram("activation_layer2", layer_2)
	# Output layer with linear activation
		out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
		tf.summary.histogram("activation_output", out_layer)
	
	return out_layer

# Store layers weight & bias

weights = {
	'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
	'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
	'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
	'b1': tf.Variable(tf.random_normal([n_hidden_1])),
	'b2': tf.Variable(tf.random_normal([n_hidden_2])),
	'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = multilayer_perceptron(x, weights, biases)

# Define loss and optimizer
with tf.name_scope("loss"):
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
	training_cost_sum = tf.summary.scalar("loss",cost)
	tf.summary.histogram("loss",cost)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Test model
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
# Calculate accuracy
with tf.name_scope("accuracy"):	
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
	training_accuracy_sum = tf.summary.scalar('accuracy', accuracy)
	tf.summary.histogram("accuracy",accuracy)


# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
saver = tf.train.Saver()
with tf.Session() as sess:
	sess.run(init)
	saver.save(sess,MODEL_PATH,global_step=1000)
	
	writer = tf.summary.FileWriter(LOG_PATH)
	merged_summary_op = tf.summary.merge_all()
	writer.add_graph(sess.graph)
	# Training cycle
	for epoch in range(training_epochs):
		avg_cost = 0.
		total_batch = int(len(data_train)/batch_size)
		X_batches = np.array_split(data_train, total_batch)
		Y_batches = np.array_split(onehot_data_label, total_batch)
		# Loop over all batches
		for i in range(total_batch):
			batch_x, batch_y = X_batches[i], Y_batches[i]
			# Run optimization 
			_,loss1, c = sess.run([optimizer, cost,merged_summary_op], feed_dict={x: batch_x,y: batch_y})
			writer.add_summary(c,epoch * total_batch + avg_cost)
			# Compute average loss
			avg_cost+=loss1
		# Display logs per epoch step
		if epoch % display_step == 0:
			print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))
	print("Optimization Finished!")


	print("Accuracy test:", accuracy.eval({x: data_test, y: onehot_test_labels}))
	print('Accuracy train:',accuracy.eval({x:data_train, y:onehot_data_label}))
	global result

	#Write the results of the classification in a file
	result = tf.argmax(pred, 1).eval({x: data_test, y: onehot_test_labels})
	le = preprocessing.LabelEncoder()
	le=le.fit(liste_of_labels_test)
	LabelEncoder()
	le.transform(liste_of_labels_test)
	liste=list(le.inverse_transform(result))
	for e in liste:
		fileOut.write(e)
		fileOut.write("\n")		
