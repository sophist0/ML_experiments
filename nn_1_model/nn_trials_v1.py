#####################################################################################################################################
# Adapted in part from kfold cross validation
# https://stackoverflow.com/questions/38164798/does-tensorflow-have-cross-validation-implemented-for-its-users/46589591#46589591
#
# Logistic Regression Trials V2
#####################################################################################################################################

from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
import random
import pandas as pd
import train_f2 as gf
import matplotlib.pyplot as plt

# construct features
CSV_COLUMN_NAMES = ['survival','pclass','sex','child','sibling/spouse','parent']
tmat = gf.get_features()	
train_tmp = pd.DataFrame(data=tmat, index=range(1,len(tmat)+1), columns=CSV_COLUMN_NAMES)
ivec = range(train_tmp.shape[0])

# Parameters
pt = 0.8						# percent training

#lr = 0.02						# learning rate
#lr = 0.05						# learning rate
#lr = 0.1						# learning rate
#lr = 0.2						# learning rate
lr = 0.5						# learning rate
#lr = 1							# learning rate

l_iter = 801						# number of learning iterations

hvec = [2,4,8,16,32]					# number of hidden nodes
num_labels = 1						# number of labels: survived
trials = 40

data_size = int(pt*train_tmp.shape[0])
test_size = train_tmp.shape[0] - data_size -1
num_f = train_tmp.shape[1]-1 # number of features, minus 1 because survival col is deleted

# new TF graph

#####################################################################################################
# NN Model 1 hidden layer
#####################################################################################################
# Variables.
num_input = 5
num_hidden = tf.Variable(2)

#num_hidden = tf.cast(num_hidden, dtype=tf.int32)
num_labels = 1

weights_1 = tf.truncated_normal(shape=[num_input, num_hidden], dtype=tf.float64)
biases_1 = tf.zeros(shape=[num_hidden], dtype=tf.float64)

weights_1 = tf.Variable(weights_1, validate_shape=False)
biases_1 = tf.Variable(biases_1, validate_shape=False)
   
weights_2 = tf.truncated_normal([num_hidden, num_labels], dtype=tf.float64)
biases_2 = tf.zeros([num_labels], dtype=tf.float64)
weights_2 = tf.Variable(weights_2, validate_shape=False)
biases_2 = tf.Variable(biases_2, validate_shape=False)

tf_train_dataset = tf.placeholder(tf.float64,[data_size,num_f])
tf_train_labels = tf.placeholder(tf.float64,[data_size,num_labels])

test_dataset = tf.placeholder(tf.float64,[test_size,num_f])

# Training computation.
out_1 = tf.nn.relu_layer(tf_train_dataset, weights=weights_1, biases=biases_1) 
logits = tf.matmul(out_1, weights_2) + biases_2

one = tf.constant(1, dtype=tf.float64)

# logistic function giving the probability of being label 1, ie survive
l_prob = tf.truediv(one, tf.add(one,tf.exp(-logits)))

# logistic loss function
c1 = tf.matmul(tf.transpose(tf_train_labels),tf.log(l_prob))
c2 = tf.matmul(tf.transpose(tf.subtract(one,tf_train_labels)),tf.log(tf.subtract(one,l_prob)))
loss = tf.multiply(tf.constant(-1/float(data_size), dtype=tf.float64), tf.add(c1,c2))

# Optimizer.
optimizer = tf.train.GradientDescentOptimizer(0.25).minimize(loss)

#####################################################################################
out_test = tf.nn.relu_layer(test_dataset, weights=weights_1, biases=biases_1) 
test_prediction = tf.matmul(out_test, weights_2) + biases_2
#####################################################################################

init = tf.global_variables_initializer()

def th_pred(pvec):
	# threshold the predictions
	for x in range(len(pvec)):
		if pvec[x] <= 0.5:
			pvec[x] = 0
		else:
			pvec[x] = 1
	return pvec

def get_acc(pred_vec, label_vec):
	c = 0
	for x in range(len(pred_vec)):
		if pred_vec[x] == label_vec[x]:
			c += 1
	acc = 100 * c/len(pred_vec)
	return acc

def run_train(session, train_x, train_y, j, h_val):
	print("\nStart training: ",j+1)
	print()
  	session.run(init)
	train_y = train_y.values
	train_y = np.reshape(train_y,(len(train_y),1))

  	for i in range(l_iter):
		_, l = session.run([optimizer, loss], feed_dict={tf_train_dataset: train_x, tf_train_labels: train_y, num_hidden: h_val})
		if i % 100 == 0:
			print("Train iteration=%d, h_val=%f, loss=%f" % (i, h_val, l))

def run_trials(session):

	rmat = []
	smat = []
	for h_val in hvec:
	  	results = []
		sex_results = []
		for j in range(trials):

			random.shuffle(ivec)
			train = train_tmp.iloc[ivec[0:data_size]]
			test = train_tmp.iloc[ivec[data_size:-1]]

			train_x, train_y = train, train.pop('survival')
			test_x, test_y = test, test.pop('survival')

			run_train(session, train_x, train_y, j, h_val)

			test_p = session.run(test_prediction, feed_dict={test_dataset: test_x})
			test_p2 = th_pred(test_p)
			test_y = test_y.values

			accuracy = get_acc(test_p2, test_y)
			results.append(accuracy)

			# model assuming all women survive
			svec = test_x.pop('sex')
			svec = th_pred(svec.values)
			sex_results.append(get_acc(svec, test_y))

		rmat.append(results)
		smat.append(sex_results)

 	return rmat, smat

with tf.Session() as session:
	rmat, smat = run_trials(session)

	rmat = np.asarray(rmat)
	smat = np.asarray(smat)
	rvec = rmat.mean(axis=1)
	svec = smat.mean(axis=1)

	print()
	print('Regulation Weight: ###############################################')
	print()
 	print(hvec)
	print()
	print('Logistic Regression: ###############################################')
	print()
 	print(rvec)
	print()
	print('Women Survive: #####################################################')
	print()
 	print(svec)
	print()
	print('###################################################################')
	print()

	plt.plot(hvec,rvec,label="Logistic Regression")
	plt.plot(hvec,svec,label="Women Survive")
	plt.legend()
	plt.xlabel("Regulator Weight")
	plt.ylabel("Accuracy")
	plt.show()


