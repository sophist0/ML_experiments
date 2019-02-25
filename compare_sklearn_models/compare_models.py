#!/usr/bin/env python

##################################################################################
# To speedup the run time reduce ptrials and trials by at least a factor of 10
##################################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
import utility as ut
import csv

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn import model_selection
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

##############################################################

tn = 50 # number of the negative class to test
#tn = 100 # number of the negative class to test

ptrials = 10
trials = 100

#ptrials = 100
#trials = 1000
ROC = True

##############################################################
# Transform dataset
ut.transform()
##############################################################


df = pd.read_csv("train_2.csv", sep=',')
df = df.reset_index(drop=True)

tcol = list(df.columns)
tcol = tcol[1:len(tcol)]
print
print "Original data columns: ", tcol
print

# Normalize feature columns
df[tcol] = (df[tcol]- df[tcol].mean()) / df[tcol].std()

##################################################
# construct training set and test set
df_yes = df[df['survived']== 1]
df_no = df[df['survived']== 0]
nyes_1 = len(df_yes)

print
print "Total observations: ", len(df)
print "Number of no labels: ", len(df_no)
print "Number of yes labels before oversampling: ", nyes_1
print

####################################################

# optimize parameters for NNC

prange = [(0,5),(6,12),(2,6)]
nnc_param = ut.grid_opt(tcol, 'nnc', prange, ptrials, df_yes, df_no, tn)
print
print nnc_param

prange = [(1,20)]
svc_param = ut.grid_opt(tcol, 'svc', prange, ptrials, df_yes, df_no, tn)
print
print svc_param
print

prange = [(1,5),(1,5)]
rfc_param = ut.grid_opt(tcol, 'rfc', prange, ptrials, df_yes, df_no, tn)
print
print rfc_param
print

prange = [(1,20),(1,6)]
knn_param = ut.grid_opt(tcol, 'knn', prange, ptrials, df_yes, df_no, tn)
print
print knn_param
print

####################################################

LOG_m = []
NNC_m = []
SVC_m = []
RFC_m = []
KNN_m = []
WS_m = []

LOG_a = []
NNC_a = []
SVC_a = []
RFC_a = []
KNN_a = []
WS_a = []

rvec = [x for x in range(trials)]
for rn in rvec:

	print
	print 'Iteration: ', rn

	train_b, test_b = train_test_split(df_yes, test_size=tn, random_state=rn)
	train_n, test_n = train_test_split(df_no, test_size=int(len(df_no)*tn/float(nyes_1)), train_size=(nyes_1-tn), random_state=rn)

	df_test = test_b.append(test_n)
	df_train = train_b.append(train_n)

	if rn == 0:
		print "Sizes of the training data"
		print
		print "train_b: ", len(train_b)
		print "train_n: ", len(train_n)
		print

		print "Sizes of unbalanced testing data"
		print
		print "test_b: ", len(test_b)
		print "test_n", len(test_n)
		print

	# shuffle the test and training sets
	df_test = df_test.sample(frac=1, random_state=rn).reset_index(drop=True)
	df_train = df_train.sample(frac=1, random_state=rn).reset_index(drop=True)

	#print
	#print "##############################################################"
	#print "Logistic Classifier"
	#print "##############################################################"
	#print

	lr = LogisticRegression(solver='liblinear', max_iter=100, random_state=rn)
	lr.fit(df_train[tcol], df_train['survived'])

	acc = lr.score(df_test[tcol],df_test['survived'])
	lr_pred = lr.predict(df_test[tcol])
	lr_cm = confusion_matrix(df_test['survived'],lr_pred)
	lr_pt = (lr_cm[0][1] / float(len(test_n)) , lr_cm[1][1] / float(len(test_b)))
	LOG_m.append(lr_pt)
	LOG_a.append(acc)

	#print "##############################################################"
	#print "ANN Classifier"
	#print "##############################################################"
	#print

	nnc = MLPClassifier(solver='lbfgs', alpha=nnc_param[0], hidden_layer_sizes=(nnc_param[1],nnc_param[2]), random_state=rn) # good

	nnc.fit(df_train[tcol],df_train['survived'])
	acc = nnc.score(df_test[tcol],df_test['survived'])
	nnc_pred = nnc.predict(df_test[tcol])
	nnc_cm = confusion_matrix(df_test['survived'],nnc_pred)
	
	nnc_pt = (nnc_cm[0][1] / float(len(test_n)) , nnc_cm[1][1] / float(len(test_b)))
	NNC_m.append(nnc_pt)
	NNC_a.append(acc)

	#print "##############################################################"
	#print "SVM Classifier"
	#print "##############################################################"
	#print

	svc_c = SVC(gamma='scale', random_state=rn, kernel='rbf', shrinking=False, C=svc_param[0])

	svc_c.fit(df_train[tcol], df_train['survived'])
	acc = svc_c.score(df_test[tcol], df_test['survived'])
	svc_pred = svc_c.predict(df_test[tcol])
	svc_cm = confusion_matrix(df_test['survived'], svc_pred)

	svc_pt = (svc_cm[0][1] / float(len(test_n)) , svc_cm[1][1] / float(len(test_b)))
	SVC_m.append(svc_pt)
	SVC_a.append(acc)

	#print "##############################################################"
	#print "Random Forest Classifier"
	#print "##############################################################"
	#print

	rfc = RandomForestClassifier(n_estimators=100, n_jobs=4, criterion='entropy', max_features=rfc_param[0], max_depth=rfc_param[1], random_state=rn)

	rfc.fit(df_train[tcol], df_train['survived'])

	acc = rfc.score(df_test[tcol], df_test['survived'])
	rfc_pred = rfc.predict(df_test[tcol])
	rfc_cm = confusion_matrix(df_test['survived'], rfc_pred)

	rfc_pt = (rfc_cm[0][1] / float(len(test_n)) , rfc_cm[1][1] / float(len(test_b)))
	RFC_m.append(rfc_pt)
	RFC_a.append(acc)

	#print "##############################################################"
	#print "KNN Classifier"
	#print "##############################################################"
	#print

	knn_c = KNeighborsClassifier(n_neighbors=knn_param[0], n_jobs=4, p=knn_param[1])
	knn_c.fit(df_train[tcol], df_train['survived'])
	acc = knn_c.score(df_test[tcol], df_test['survived'])
	knn_c_pred = knn_c.predict(df_test[tcol])
	knn_cm = confusion_matrix(df_test['survived'], knn_c_pred)

	knn_pt = (knn_cm[0][1] / float(len(test_n)) , knn_cm[1][1] / float(len(test_b)))
	KNN_m.append(knn_pt)
	KNN_a.append(acc)

	#print "##############################################################"
	#print "Women Survive Classifier"
	#print "##############################################################"
	#print

	tmp = df_test['sex'].copy()
	c = 0
	for x in range(len(df_test['survived'])):

		# un-normalize sex
		if tmp[x] <= 0:
			tmp[x] = 0
		else:
			tmp[x] = 1 

		if df_test['survived'][x] == tmp[x]:
			c+=1

	acc = c/float(len(df_test['survived']))
	ws_cm = confusion_matrix(df_test['survived'], tmp)
	ws_pt = (ws_cm[0][1] / float(len(test_n)) , ws_cm[1][1] / float(len(test_b)))

	WS_m.append(ws_pt)
	WS_a.append(acc)

lvec = ['Logistic Regression', 'Artificial Neural Network', 'Support Vector Machine', 'Random Forest', 'KNN', 'Women Survive']

#############################################################
# Compare models in ROC space
#############################################################

CM = [LOG_m, NNC_m, SVC_m, RFC_m, KNN_m, WS_m]
ACC = [LOG_a, NNC_a, SVC_a, RFC_a, KNN_a, WS_a]
cvec = ['b', 'r', 'g', 'm', 'k', 'y']

pvec = []
for x in range(len(CM)):

	rtuple = ut.ave_std(CM[x])
	pvec.append(rtuple)
	#return (xave,yave,xstd,ystd)

print "##############################################################"
print
for x in range(len(ACC)):
	tmp = np.asarray(ACC[x])
	ave = tmp.mean()
	print lvec[x] + ': ' + str(ave)

print
print "##############################################################"

print
print pvec
print
if ROC:

	mdpi = 96
	plt.figure(figsize=(1200/mdpi,1200/mdpi), dpi=mdpi)

	# Plot ROC
	for x in range(len(pvec)):

		plt.errorbar(pvec[x][0], pvec[x][1], xerr=pvec[x][2], yerr=pvec[x][3], c=cvec[x], label=lvec[x], marker='.', ms=20)

	plt.plot([0,1], [0,1], ls='--', c='r', lw=2)
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.xlim((0,1))
	plt.ylim((0,1))
	plt.legend(loc=4)
	plt.savefig('compare_models.png')
	plt.show()

########################################################################
# Save model parameters and figure
########################################################################

with open('model_param.csv', 'a') as csvFile:
    	writer = csv.writer(csvFile)
    	writer.writerow(["nnc param: "] + nnc_param)
    	writer.writerow(["svc param: "] + svc_param)
    	writer.writerow(["rfc param: "] + rfc_param)
    	writer.writerow(["knn param:"] + knn_param)

	for x in range(len(ACC)):
		tmp = np.asarray(ACC[x])
		ave = tmp.mean()
		writer.writerow([lvec[x] + ': ', str(ave)])

csvFile.close()

