## ============================================================================================================================
## This code was modified from Google's Tensor Flow example code demoing the use of premade estimators: premade_estimator.py
## I use part of the data set from Kaggle Titantic challenge to predict passenger survival.
##
## Version 1: Linear Regression, no feature selection with penalties or feature normalization, does not perform much better then predicting women survive
## ============================================================================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import random
import pandas as pd
import train_f as gf

CSV_COLUMN_NAMES = ['survival','pclass','sex','child','sibling/spouse','parent']

def main(argv):

	# construct features
	tmat = gf.get_features()	
	train_tmp = pd.DataFrame(data=tmat, index=range(1,len(tmat)+1), columns=CSV_COLUMN_NAMES)
	
	# get the data from csv.
	#train_tmp = pd.read_csv('train_2.csv',names=CSV_COLUMN_NAMES,header=0)

	ivec = range(train_tmp.shape[0])
	random.shuffle(ivec)

	# split out the training set	
	ps = 0.8
	l = int(ps*train_tmp.shape[0])
	
	train = train_tmp.iloc[ivec[0:l]]
	test = train_tmp.iloc[ivec[l:-1]]

	train_x, train_y = train, train.pop('survival')
	test_x, test_y = test, test.pop('survival')

	# Feature columns describe how to use the input.
	my_feature_columns = []
	for key in train_x.keys():
		my_feature_columns.append(tf.feature_column.numeric_column(key=key))

	# Setup linear regression model
	batch_size = 32
        buffer_size = 1000 # maximum number of elements in the dataset that are shuffled at once
	model = tf.estimator.LinearRegressor(feature_columns=my_feature_columns)


	"""Builds, trains, and evaluates the model."""
	model.train(input_fn=lambda:input_func(train_x,train_y,batch_size,buffer_size), steps=2500)

	eval_result = model.evaluate(input_fn=lambda:input_func_test(test_x,test_y,batch_size=1))

	print()
	print("#########################################################################")
	print()
	print('Test set average loss: ',eval_result['average_loss'])
	print()

	# get predictions as a generator
	pred = model.predict(input_fn=lambda:input_func_eval(test_x, None,batch_size=1))

	pvec = []
	for p in list(pred):
		pvec.append(int(round(p["predictions"][0])))

	test_labels = test_y.values
	c = 0
	for x in range(len(pvec)):
		if pvec[x] == test_labels[x]:
			c += 1

	acc = 100 * c/len(pvec)
	print("Test set accuracy: ", acc)
	print()
	print("#########################################################################")
	print()	

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
