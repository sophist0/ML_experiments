## ============================================================================================================================
## I use part of the data set from Kaggle Titantic challenge to predict passenger survival.
##
## Version 3: Logistic Regression, accuracy about the same as assuming all women survive, regulation weight selected adhoc.
## ============================================================================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import random
import pandas as pd
import train_f2 as gf

CSV_COLUMN_NAMES = ['survival','pclass','sex','child','sibling/spouse','parent']

def main(argv):

	# construct features
	tmat = gf.get_features()	
	train_tmp = pd.DataFrame(data=tmat, index=range(1,len(tmat)+1), columns=CSV_COLUMN_NAMES)
	
	ivec = range(train_tmp.shape[0])
	random.shuffle(ivec)

	# split out the training set	
	ps = 0.8
	num_train = int(ps*train_tmp.shape[0])
	
	train = train_tmp.iloc[ivec[0:num_train]]
	test = train_tmp.iloc[ivec[num_train:-1]]

	num_labels = 1
	train_x, train_y = train, train.pop('survival')
	test_x, test_y = test, test.pop('survival')

	# Setup logistic regression model
	graph = tf.Graph()
	with graph.as_default():

		tf_train_dataset = tf.constant(train_x)
		tf_train_labels = tf.constant(train_y)

		data_col = tf_train_dataset.shape[1].value
		weights = tf.Variable(tf.truncated_normal([data_col, num_labels]))
		biases = tf.Variable(tf.zeros([num_labels]))

		weights = tf.cast(weights, dtype=tf.float64)
		biases = tf.cast(biases, dtype=tf.float64)

		tf_train_labels = tf.reshape(tf_train_labels,[num_train,1])
		tf_train_labels = tf.cast(tf_train_labels, dtype=tf.float64)
		
		logits = tf.matmul(tf_train_dataset, weights) + biases

		one = tf.constant(1, dtype=tf.float64)

		# logistic function giving the probability of being label 1, ie survive
		l_prob = tf.truediv(one, tf.add(one,tf.exp(-logits)))

		# logistic loss function
		c1 = tf.matmul(tf.transpose(tf_train_labels),tf.log(l_prob))
		c2 = tf.matmul(tf.transpose(tf.subtract(one,tf_train_labels)),tf.log(tf.subtract(one,l_prob)))
		loss = tf.multiply(tf.constant(-1/float(num_train), dtype=tf.float64), tf.add(c1,c2))

		# add regulation term
		reg_term = tf.multiply(tf.constant(0.01, dtype=tf.float64),tf.matmul(tf.transpose(weights),weights))
		loss = tf.add(loss,reg_term)

		# Optimizer.
		optimizer = tf.train.GradientDescentOptimizer(0.25).minimize(loss)
	  
		train_prediction = l_prob

		tf_test_dataset = tf.constant(test_x)
		tf_test_labels = tf.constant(test_y)
		test_prediction = tf.matmul(tf_test_dataset, weights) + biases


	num_steps = 801
	with tf.Session(graph=graph) as session:
		tf.global_variables_initializer().run()
	  	print('Initialized')
	  	for step in range(num_steps):
	    		_, l, predictions = session.run([optimizer, loss, train_prediction])

	    		if (step % 100 == 0):
	      			print('Loss at step %d: %f' % (step, l))

		# Requires the session to be defined, "with...", to evaluate
		test_labels = test_y.values
		test_p = th_pred(test_prediction.eval())
		model_acc = get_acc(test_p, test_y.values)

	print()
	print("#########################################################################")
	print()

	svec = test_x.pop('sex')
	slist = svec.tolist()
	slist = un_norm(slist)
	sex_acc = get_acc(slist, test_labels)
	
	print("Test set model accuracy: ", model_acc)
	print()
	print("Test set female model accuracy: ", sex_acc)
	print()
	print("#########################################################################")
	print()


def th_pred(pvec):

	# threshold the predictions
	for x in range(len(pvec)):
		if pvec[x] <= 0.5:
			pvec[x] = 0
		else:
			pvec[x] = 1

	return pvec

def un_norm(svec):

	# hack to recover un normalized sex labels as 0, 1
	m = max(svec)
	for x in range(len(svec)):
		if svec[x] == m:
			svec[x] = 1
		else:
			svec[x] = 0

	return svec
	

def get_acc(pred_vec, label_vec):

	c = 0
	for x in range(len(pred_vec)):
		#print(str(pred_vec[x]) + ', ' + str(label_vec[x]))
		if pred_vec[x] == label_vec[x]:
			c += 1

	acc = 100 * c/len(pred_vec)

	return acc


def input_func(features,labels,batch_size,buffer_size):

     	# Convert the inputs to a Dataset.
     	dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

     	# Shuffle, repeat, and batch the examples.
   	dataset = dataset.shuffle(buffer_size).repeat().batch(batch_size)

	return dataset

def input_func_test(features,labels,batch_size):

     	# Convert the inputs to a Dataset.
     	dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

     	# Shuffle, repeat, and batch the examples.
   	dataset = dataset.batch(batch_size)

	return dataset

def input_func_eval(features, labels, batch_size):
    """An input function for evaluation or prediction"""
    features=dict(features)
    if labels is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, labels)

    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    # Return the dataset.
    return dataset

if __name__ == "__main__":

  	# The Estimator periodically generates "INFO" logs; make these logs visible.
  	# tf.logging.set_verbosity(tf.logging.INFO)
  	tf.app.run(main=main)
