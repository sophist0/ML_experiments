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
lr = 0.1						# learning rate
#lr = 0.2						# learning rate
#lr = 0.5						# learning rate
#lr = 1							# learning rate

l_iter = 501						# number of learning iterations
#rw = 0.01						# weight of the regulator
rw_vec = [x/float(100) for x in range(1,27,5)]		# vector of regulator weights
rw_vec = [0] + rw_vec
num_labels = 1						# number of labels: survived
trials = 20

data_size = int(pt*train_tmp.shape[0])
test_size = train_tmp.shape[0] - data_size -1
num_f = train_tmp.shape[1]-1 # number of features, minus 1 because survival col is deleted

# new TF graph
tf_train_dataset = tf.placeholder(tf.float64,[data_size,num_f])
tf_train_labels = tf.placeholder(tf.float64,[data_size,num_labels])

test_dataset = tf.placeholder(tf.float64,[test_size,num_f])
test_labels = tf.placeholder(tf.float64,[test_size,num_labels])

weights = tf.Variable(tf.truncated_normal([num_f, num_labels]))
biases = tf.Variable(tf.zeros([num_labels]))
rw = tf.Variable(1)

tf_train_labels = tf.cast(tf_train_labels, dtype=tf.float64)
weights = tf.cast(weights, dtype=tf.float64)
biases = tf.cast(biases, dtype=tf.float64)
rw = tf.cast(rw, dtype=tf.float64)

logits = tf.matmul(tf_train_dataset, weights) + biases
one = tf.constant(1, dtype=tf.float64)

# logistic function giving the probability of being label 1, ie survive
l_prob = tf.truediv(one, tf.add(one,tf.exp(-logits)))

# logistic loss function
c1 = tf.matmul(tf.transpose(tf_train_labels),tf.log(l_prob))
c2 = tf.matmul(tf.transpose(tf.subtract(one,tf_train_labels)),tf.log(tf.subtract(one,l_prob)))
loss = tf.multiply(tf.constant(-1/float(data_size), dtype=tf.float64), tf.add(c1,c2))

print()
print(tf.matmul(tf.transpose(weights),weights))
print(rw)
print()
# add regulation term
reg_term = tf.multiply(rw,tf.matmul(tf.transpose(weights),weights))
loss = tf.add(loss,reg_term)

# Optimizer.
optimizer = tf.train.GradientDescentOptimizer(lr).minimize(loss)
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
		#print(str(pred_vec[x]) + ', ' + str(label_vec[x]))
		if pred_vec[x] == label_vec[x]:
			c += 1
	acc = 100 * c/len(pred_vec)
	return acc

test_prediction = tf.matmul(test_dataset, weights) + biases

def run_train(session, train_x, train_y, j, rw_val):
	print("\nStart training: ",j+1)
	print()
  	session.run(init)
	train_y = train_y.values
	train_y = np.reshape(train_y,(len(train_y),1))

  	for i in range(l_iter):
		_, l = session.run([optimizer, loss], feed_dict={tf_train_dataset: train_x, tf_train_labels: train_y, rw: rw_val})
		if i % 100 == 0:
			print("Train iteration=%d, rw=%f, loss=%f" % (i, rw_val, l))

def run_trials(session):

	rmat = []
	smat = []
	for rw_val in rw_vec:
	  	results = []
		sex_results = []
		for j in range(trials):

			random.shuffle(ivec)
			train = train_tmp.iloc[ivec[0:data_size]]
			test = train_tmp.iloc[ivec[data_size:-1]]

			train_x, train_y = train, train.pop('survival')
			test_x, test_y = test, test.pop('survival')

			run_train(session, train_x, train_y, j, rw_val)

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

	#print()
	#print(rmat)
	#print()
	#print(kill)
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
 	print(rw_vec)
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

	plt.plot(rw_vec,rvec,label="Logistic Regression")
	plt.plot(rw_vec,svec,label="Women Survive")
	plt.legend()
	plt.xlabel("Regulator Weight")
	plt.ylabel("Accuracy")
	plt.show()


