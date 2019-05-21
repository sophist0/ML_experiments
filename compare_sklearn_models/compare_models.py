#!/usr/bin/env python

##################################################################################
# To speedup the run time reduce ptrials and trials by at least a factor of 10
##################################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
import utility_15 as ut
import csv
import math
import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn import model_selection
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

##############################################################

wd = False		# meta models with original data
PCA_m = False
PCA_f = True
REDUCED_m = True	# reduced training data (MALE)
REDUCED_f = False	# reduced training data (FEMALE)
rn = 1
trials = 10
ROC = True
##############################################################
# Transform dataset
load_f = "data/train.csv"
save_f = "data/train_3.csv"
df_train = pd.read_csv(save_f, sep=',')

##################################################
# Do we need ptm or ptf? From training data
##################################################
onedie, onesur, tickets = ut.train_transform(load_f,save_f)
[df_train_m, df_train_f, n_ave_m, n_ave_f, n_std_m, n_std_f, mu_train, fu_train, ptm, ptf] = ut.class_easy(save_f,REDUCED_m,REDUCED_f)

tcol_m = list(df_train_m.columns)
tcol_f = list(df_train_f.columns)
tcol_m.remove('survived') # don't include survival column
tcol_m.remove('pid')
tcol_f.remove('survived')
tcol_f.remove('pid')

print
print tcol_m
print tcol_f
print

##################################################
# Load testing and ground truth sets

df_test = pd.read_csv("data/test_mod.csv", sep=',')

# load ground truth
df_gnd = pd.read_csv("data/titanic_ground.csv", sep=',')

#df_test_gnd = pd.merge(df_test, df_gnd[['Name','Survived','Ticket',]], how='inner', on=['Name','Ticket'])
df_test_gnd = pd.merge(df_test, df_gnd, how='inner', on=['Name','Ticket'])
df_test_gnd = df_test_gnd[['PassengerId','Survived','Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked']]

load_f = "data/test_gnd.csv"
save_f = "data/test_gnd_3.csv"
df_test_gnd.to_csv(load_f,sep=',', index=False)

############################################################
# new version
############################################################
ut.test_transform(load_f,save_f, onedie, onesur, tickets)

# Still ordered (test set)
df_tg = pd.read_csv(save_f, sep=',')
df_tm = df_tg[df_tg['sex']==0]
df_tf = df_tg[df_tg['sex']==1]
test_bm = df_tm[df_tm['survived']== 1]
test_nm = df_tm[df_tm['survived']== 0]
test_bf = df_tf[df_tf['survived']== 1]
test_nf = df_tf[df_tf['survived']== 0]

ivec_m = list(df_tm['pid'])
ivec_f = list(df_tf['pid'])

##########################################
# survived in test set
##########################################
svec_m = list(df_tm['survived'])
svec_f = list(df_tf['survived'])

[df_test_m, df_test_f, mu_test, fu_test, pvec_m, pvec_f] = ut.class_easy_test(df_tg, tcol_m, tcol_f)

df_test_m[tcol_m] = (df_test_m[tcol_m]- n_ave_m) / n_std_m
df_test_f[tcol_f] = (df_test_f[tcol_f]- n_ave_f) / n_std_f

############################################################
# PCA
############################################################
print
print tcol_m
print tcol_f
if PCA_m:
	# training set
	cm = ["C"+str(x+1) for x in range(len(tcol_m))]
	tmp_m1 = df_train_m[['pid','survived']]
	tmp_m2, eigvec_m = ut.pca_trans(df_train_m[tcol_m], [], cm)
	df_train_m = tmp_m1.join(tmp_m2, how='left')

	# test set
	tmp_m1 = df_test_m[['pid','survived']]
	tmp_m2, eigvec_m = ut.pca_trans(df_test_m[tcol_m], eigvec_m, cm)
	df_test_m = tmp_m1.join(tmp_m2, how='left')

if PCA_f:
	# training set
	cf = ["C"+str(x+1) for x in range(len(tcol_f))]
	tmp_f1 = df_train_f[['pid','survived']]
	tmp_f2, eigvec_f = ut.pca_trans(df_train_f[tcol_f], [], cf)
	df_train_f = tmp_f1.join(tmp_f2, how='left')

	# test set
	tmp_f1 = df_test_f[['pid','survived']]
	tmp_f2, eigvec_f = ut.pca_trans(df_test_f[tcol_f], eigvec_f, cf)
	df_test_f = tmp_f1.join(tmp_f2, how='left')

####################################################

tcol_m = list(df_train_m.columns)
tcol_f = list(df_train_f.columns)
tcol_m.remove('survived')
tcol_m.remove('pid')
tcol_f.remove('survived')
tcol_f.remove('pid')
###########################################################################
# Load optimized features and parameters for order 1 model
###########################################################################

if PCA_m and REDUCED_m:
	path_m = "fsets_pca_red/"
elif not PCA_m and REDUCED_m:
	path_m = "fsets_red/"
elif PCA_m and not REDUCED_m:
	path_m = "fsets_pca/"
else:
	path_m = "fsets/"

if PCA_f and REDUCED_f:
	path_f = "fsets_pca_red/"
elif not PCA_f and REDUCED_f:
	path_f = "fsets_red/"
elif PCA_f and not REDUCED_f:
	path_f = "fsets_pca/"
else:
	path_f = "fsets/"

lr_dm = pickle.load(open(path_m+"lr_m.p","rb"))
lr_df = pickle.load(open(path_f+"lr_f.p","rb"))

nnc_dm = pickle.load(open(path_m+"nnc_m.p","rb"))
nnc_df = pickle.load(open(path_f+"nnc_f.p","rb"))

svc_dm = pickle.load(open(path_m+"svc_m.p","rb"))
svc_df = pickle.load(open(path_f+"svc_f.p","rb"))

rfc_dm = pickle.load(open(path_m+"rfc_m.p","rb"))
rfc_df = pickle.load(open(path_f+"rfc_f.p","rb"))

knn_dm = pickle.load(open(path_m+"knn_m.p","rb"))
knn_df = pickle.load(open(path_f+"knn_f.p","rb"))

gnb_dm = pickle.load(open(path_m+"gnb_m.p","rb"))
gnb_df = pickle.load(open(path_f+"gnb_f.p","rb"))

# first order model features
#################################################################
# features are chosen in a somewhat adhoc manner
#
# next time I train the features I need to get rid of the 
# features that have all the same value. After retraining
# update feature selection based on recomputed accuracy.
#################################################################
m1_param_m = {}
m1_param_f = {}

####################################
# REDUCED ??????
####################################
if PCA_m and REDUCED_m:

	m1_param_m['lr_f'] = lr_dm["features"][0:6]
	m1_param_m['nnc_f'] = nnc_dm["features"][0:5]
	m1_param_m['svc_f'] = svc_dm["features"][0:7]
	m1_param_m['rfc_f'] = rfc_dm["features"][0:4]
	m1_param_m['knn_f'] = knn_dm["features"][0:6]
	m1_param_m['gnb_f'] = gnb_dm["features"][0:6]

	# first order model parameters, matching the number of features
	m1_param_m['nnc'] = nnc_dm["parameters"][4]
	m1_param_m['svc'] = svc_dm["parameters"][6]
	m1_param_m['rfc'] = rfc_dm["parameters"][3]
	m1_param_m['knn'] = knn_dm["parameters"][5]

elif not PCA_m and REDUCED_m:

	#print
	#print "#################################################"
	#print
	#print gnb_dm["accuracy"]
	#print
	#print gnb_df["accuracy"]
	#print
	#print kill

	m1_param_m['lr_f'] = lr_dm["features"][0:3]
	m1_param_m['nnc_f'] = nnc_dm["features"][0:7]
	m1_param_m['svc_f'] = svc_dm["features"][0:7]
	m1_param_m['rfc_f'] = rfc_dm["features"][0:7]
	m1_param_m['knn_f'] = knn_dm["features"][0:11]
	m1_param_m['gnb_f'] = gnb_dm["features"][0:8] 	

	# first order model parameters, matching the number of features
	m1_param_m['nnc'] = nnc_dm["parameters"][6]
	m1_param_m['svc'] = svc_dm["parameters"][6]
	m1_param_m['rfc'] = rfc_dm["parameters"][6]
	m1_param_m['knn'] = knn_dm["parameters"][10]

elif PCA_m and not REDUCED_m:

	m1_param_m['lr_f'] = lr_dm["features"][0:5]
	m1_param_m['nnc_f'] = nnc_dm["features"][0:3]
	m1_param_m['svc_f'] = svc_dm["features"][0:4]
	m1_param_m['rfc_f'] = rfc_dm["features"][0:2]
	m1_param_m['knn_f'] = knn_dm["features"][0:6]
	m1_param_m['gnb_f'] = gnb_dm["features"][0:2]

	# first order model parameters, matching the number of features
	m1_param_m['nnc'] = nnc_dm["parameters"][2]
	m1_param_m['svc'] = svc_dm["parameters"][3]
	m1_param_m['rfc'] = rfc_dm["parameters"][1]
	m1_param_m['knn'] = knn_dm["parameters"][5]

else:

	m1_param_m['lr_f'] = lr_dm["features"][0:4]
	m1_param_m['nnc_f'] = nnc_dm["features"][0:6]
	m1_param_m['svc_f'] = svc_dm["features"][0:3]
	m1_param_m['rfc_f'] = rfc_dm["features"][0:3]
	m1_param_m['knn_f'] = knn_dm["features"][0:3]
	m1_param_m['gnb_f'] = gnb_dm["features"][0:1]

	# first order model parameters, matching the number of features
	m1_param_m['nnc'] = nnc_dm["parameters"][5]
	m1_param_m['svc'] = svc_dm["parameters"][2]
	m1_param_m['rfc'] = rfc_dm["parameters"][2]
	m1_param_m['knn'] = knn_dm["parameters"][2]

if PCA_f and REDUCED_f:

	m1_param_f['lr_f'] = nnc_df["features"][0:4]
	m1_param_f['nnc_f'] = nnc_df["features"][0:6]
	m1_param_f['svc_f'] = svc_df["features"][0:2]
	m1_param_f['rfc_f'] = rfc_df["features"][0:4]
	m1_param_f['knn_f'] = knn_df["features"][0:2]
	m1_param_f['gnb_f'] = gnb_df["features"][0:1]

	m1_param_f['nnc'] = nnc_df["parameters"][5]
	m1_param_f['svc'] = svc_df["parameters"][1]
	m1_param_f['rfc'] = rfc_df["parameters"][3]
	m1_param_f['knn'] = knn_df["parameters"][1]

elif not PCA_f and REDUCED_f:

	m1_param_f['lr_f'] = lr_df["features"][0:7]
	m1_param_f['nnc_f'] = nnc_df["features"][0:6]
	m1_param_f['svc_f'] = svc_df["features"][0:4]
	m1_param_f['rfc_f'] = rfc_df["features"][0:8]
	m1_param_f['knn_f'] = knn_df["features"][0:3]
	m1_param_f['gnb_f'] = gnb_df["features"][0:3]	

	m1_param_f['nnc'] = nnc_df["parameters"][5]
	m1_param_f['svc'] = svc_df["parameters"][3]
	m1_param_f['rfc'] = rfc_df["parameters"][7]
	m1_param_f['knn'] = knn_df["parameters"][2]

elif PCA_f and not REDUCED_f:

	m1_param_f['lr_f'] = nnc_df["features"][0:6]
	m1_param_f['nnc_f'] = nnc_df["features"][0:4]
	m1_param_f['svc_f'] = svc_df["features"][0:5]
	m1_param_f['rfc_f'] = rfc_df["features"][0:4]
	m1_param_f['knn_f'] = knn_df["features"][0:6]
	m1_param_f['gnb_f'] = gnb_df["features"][0:8]

	m1_param_f['nnc'] = nnc_df["parameters"][3]
	m1_param_f['svc'] = svc_df["parameters"][4]
	m1_param_f['rfc'] = rfc_df["parameters"][3]
	m1_param_f['knn'] = knn_df["parameters"][5]

else:

	m1_param_f['lr_f'] = lr_df["features"][0:7]
	m1_param_f['nnc_f'] = nnc_df["features"][0:6]
	m1_param_f['svc_f'] = svc_df["features"][0:9]
	m1_param_f['rfc_f'] = rfc_df["features"][0:8]
	m1_param_f['knn_f'] = knn_df["features"][0:5]
	m1_param_f['gnb_f'] = gnb_df["features"][0:2]

	m1_param_f['nnc'] = nnc_df["parameters"][5]
	m1_param_f['svc'] = svc_df["parameters"][8]
	m1_param_f['rfc'] = rfc_df["parameters"][7]
	m1_param_f['knn'] = knn_df["parameters"][4]



#######################################################
# meta models
#
# The data used here is never reduced as any feature
# may be important in the meta model even if its
# not in the reduced feature set.
#######################################################

# male
# get logistic regression predictions
col = m1_param_m['lr_f']
lr = LogisticRegression(solver='liblinear', max_iter=100, random_state=rn)
lr.fit(df_train_m[col], df_train_m['survived'])
lr_pred_m1 = lr.predict(df_train_m[col])
lr_pred_m2 = lr.predict(df_test_m[col])

# get neural net predictions
col = m1_param_m['nnc_f']
nnc = MLPClassifier(solver='lbfgs', alpha=(0.0001*(math.pow(10,m1_param_m['nnc'][0]))), hidden_layer_sizes=(m1_param_m['nnc'][1],m1_param_m['nnc'][2]), random_state=rn) 
nnc.fit(df_train_m[col],df_train_m['survived'])
nnc_pred_m1 = nnc.predict(df_train_m[col])
nnc_pred_m2 = nnc.predict(df_test_m[col])

# get support vector machine predictions
col = m1_param_m['svc_f']
svc_c = SVC(gamma='scale', random_state=rn, kernel='rbf', shrinking=False, C=m1_param_m['svc'][0])
svc_c.fit(df_train_m[col], df_train_m['survived'])
svc_pred_m1 = svc_c.predict(df_train_m[col])
svc_pred_m2 = svc_c.predict(df_test_m[col])

# get random forest predictions
col = m1_param_m['rfc_f']
rfc = RandomForestClassifier(n_estimators=100, n_jobs=4, criterion='entropy', max_depth=m1_param_m['rfc'][0], random_state=rn)
rfc.fit(df_train_m[col], df_train_m['survived'])
rfc_pred_m1 = rfc.predict(df_train_m[col])
rfc_pred_m2 = rfc.predict(df_test_m[col])

# get KNN predictions
col = m1_param_m['knn_f']
knn_c = KNeighborsClassifier(n_neighbors=m1_param_m['knn'][0], n_jobs=4, p=m1_param_m['knn'][1])
knn_c.fit(df_train_m[col], df_train_m['survived'])
knn_pred_m1 = knn_c.predict(df_train_m[col])
knn_pred_m2 = knn_c.predict(df_test_m[col])

# get NB predictions
col = m1_param_m['gnb_f']
gnb_c = GaussianNB()
gnb_c.fit(df_train_m[col], df_train_m['survived'])
gnb_pred_m1 = gnb_c.predict(df_train_m[col])
gnb_pred_m2 = gnb_c.predict(df_test_m[col])

# female
# get logistic regression predictions
col = m1_param_f['lr_f']
lr = LogisticRegression(solver='liblinear', max_iter=100, random_state=rn)
lr.fit(df_train_f[col], df_train_f['survived'])
lr_pred_f1 = lr.predict(df_train_f[col])
lr_pred_f2 = lr.predict(df_test_f[col])

# get neural net predictions
col = m1_param_f['nnc_f']
nnc = MLPClassifier(solver='lbfgs', alpha=(0.0001*(math.pow(10,m1_param_f['nnc'][0]))), hidden_layer_sizes=(m1_param_f['nnc'][1],m1_param_f['nnc'][2]), random_state=rn)
nnc.fit(df_train_f[col],df_train_f['survived'])
nnc_pred_f1 = nnc.predict(df_train_f[col])
nnc_pred_f2 = nnc.predict(df_test_f[col])

# get support vector machine predictions
col = m1_param_f['svc_f']
svc_c = SVC(gamma='scale', random_state=rn, kernel='rbf', shrinking=False, C=m1_param_f['svc'][0])
svc_c.fit(df_train_f[col], df_train_f['survived'])
svc_pred_f1 = svc_c.predict(df_train_f[col])
svc_pred_f2 = svc_c.predict(df_test_f[col])

# get random forest predictions
col = m1_param_f['rfc_f']
rfc = RandomForestClassifier(n_estimators=100, n_jobs=4, criterion='entropy', max_depth=m1_param_f['rfc'][0], random_state=rn)
rfc.fit(df_train_f[col], df_train_f['survived'])
rfc_pred_f1 = rfc.predict(df_train_f[col])
rfc_pred_f2 = rfc.predict(df_test_f[col])

# get KNN predictions
col = m1_param_f['knn_f']
knn_c = KNeighborsClassifier(n_neighbors=m1_param_f['knn'][0], n_jobs=4, p=m1_param_f['knn'][1])
knn_c.fit(df_train_f[col], df_train_f['survived'])
knn_pred_f1 = knn_c.predict(df_train_f[col])
knn_pred_f2 = knn_c.predict(df_test_f[col])

# get NB predictions
col = m1_param_f['gnb_f']
gnb_c = GaussianNB()
gnb_c.fit(df_train_f[col], df_train_f['survived'])
gnb_pred_f1 = gnb_c.predict(df_train_f[col])
gnb_pred_f2 = gnb_c.predict(df_test_f[col])

# first order model predictions
train_pm = {}
train_pm['lrc'] = lr_pred_m1
train_pm['nnc'] = nnc_pred_m1
train_pm['svc'] = svc_pred_m1
train_pm['rfc'] = rfc_pred_m1
train_pm['knn'] = knn_pred_m1
train_pm['gnb'] = gnb_pred_m1

train_pf = {}
train_pf['lrc'] = lr_pred_f1
train_pf['nnc'] = nnc_pred_f1
train_pf['svc'] = svc_pred_f1
train_pf['rfc'] = rfc_pred_f1
train_pf['knn'] = knn_pred_f1
train_pf['gnb'] = gnb_pred_f1

test_pm = {}
test_pm['lrc'] = lr_pred_m2
test_pm['nnc'] = nnc_pred_m2
test_pm['svc'] = svc_pred_m2
test_pm['rfc'] = rfc_pred_m2
test_pm['knn'] = knn_pred_m2
test_pm['gnb'] = gnb_pred_m2

test_pf = {}
test_pf['lrc'] = lr_pred_f2
test_pf['nnc'] = nnc_pred_f2
test_pf['svc'] = svc_pred_f2
test_pf['rfc'] = rfc_pred_f2
test_pf['knn'] = knn_pred_f2
test_pf['gnb'] = gnb_pred_f2

pt, acc, m2_param_m = ut.construct_meta(df_train_m, df_test_m, test_bm, test_nm, train_pm, test_pm, tcol_m, wd, rn, PCA_m)
print
print pt
print acc
print m2_param_m
print

pt, acc, m2_param_f = ut.construct_meta(df_train_f, df_test_f, test_bf, test_nf, train_pf, test_pf, tcol_f, wd, rn, PCA_f)
print
print pt
print acc
print m2_param_f
print

####################################################

# male
LOG_mm = []
NNC_mm = []
SVC_mm = []
RFC_mm = []
KNN_mm = []
GNB_mm = []
WS_mm = []

LOG_mm2 = []
NNC_mm2 = []
SVC_mm2 = []
KNN_mm2 = []
GNB_mm2 = []

LOG_am = []
NNC_am = []
SVC_am = []
RFC_am = []
KNN_am = []
GNB_am = []
WS_am = []

LOG_am2 = []
NNC_am2 = []
SVC_am2 = []
KNN_am2 = []
GNB_am2 = []

# female
LOG_mf = []
NNC_mf = []
SVC_mf = []
RFC_mf = []
KNN_mf = []
GNB_mf = []
WS_mf = []

LOG_mf2 = []
NNC_mf2 = []
SVC_mf2 = []
KNN_mf2 = []
GNB_mf2 = []

LOG_af = []
NNC_af = []
SVC_af = []
RFC_af = []
KNN_af = []
GNB_af = []
WS_af = []

LOG_af2 = []
NNC_af2 = []
SVC_af2 = []
KNN_af2 = []
GNB_af2 = []

lvec = ['Logistic Regression', 'Artificial Neural Network', 'Support Vector Machine', 'Random Forest', 'KNN', 'Naive Bayes', 'Women Survive', 'Logistic Regression (meta)', 'Artificial Neural Network (meta)', 'Support Vector Machine (meta)', 'KNN (meta)', 'Naive Bayes (meta)']
lvec_acc = ['lr acc:', 'nnc acc:', 'svc acc:', 'rfc acc:', 'knn acc:', 'gnb acc:', 'ws acc:', 'lr (meta) acc:', 'nnc acc (meta):', 'svc (meta) acc:', 'knn acc (meta):', 'gnb acc (meta):']

# Looping only for the STD of the plots
rvec = [x for x in range(trials)]
for rnx in rvec:

	print
	print 'Iteration: ', rnx

	# male
	men = True
	pt1, acc1, eacc_m1, mp_m1 = ut.get_roc(df_train_m, df_test_m, test_bm, test_nm, m1_param_m, mu_test, pvec_m, svec_m, men, rnx)

	LOG_mm.append(pt1[0])
	NNC_mm.append(pt1[1])
	SVC_mm.append(pt1[2])
	RFC_mm.append(pt1[3])
	KNN_mm.append(pt1[4])
	GNB_mm.append(pt1[5])
	WS_mm.append(pt1[6])

	pt2, acc2, mp_m2 = ut.get_roc_meta(df_train_m, df_test_m, test_bm, test_nm, m2_param_m, train_pm, test_pm, tcol_m, mu_test, pvec_m, svec_m, wd, rnx, PCA_m)
	
	LOG_mm2.append(pt2[0])
	NNC_mm2.append(pt2[1])
	SVC_mm2.append(pt2[2])
	KNN_mm2.append(pt2[3])
	GNB_mm2.append(pt2[4])

	if rnx == rn:
		LOG_am.append(acc1[0])
		NNC_am.append(acc1[1])
		SVC_am.append(acc1[2])
		RFC_am.append(acc1[3])
		KNN_am.append(acc1[4])
		GNB_am.append(acc1[5])
		WS_am.append(acc1[6])

		LOG_am2.append(acc2[0])
		NNC_am2.append(acc2[1])
		SVC_am2.append(acc2[2])
		KNN_am2.append(acc2[3])
		GNB_am2.append(acc2[4])

	# female
	men = False
	pt1, acc1, eacc_f1, mp_f1 = ut.get_roc(df_train_f, df_test_f, test_bf, test_nf, m1_param_f, fu_test, pvec_f, svec_f, men, rnx)

	LOG_mf.append(pt1[0])
	NNC_mf.append(pt1[1])
	SVC_mf.append(pt1[2])
	RFC_mf.append(pt1[3])
	KNN_mf.append(pt1[4])
	GNB_mf.append(pt1[5])
	WS_mf.append(pt1[6])

	pt2, acc2, mp_f2 = ut.get_roc_meta(df_train_f, df_test_f, test_bf, test_nf, m2_param_f, train_pf, test_pf, tcol_f, fu_test, pvec_f, svec_f, wd, rnx, PCA_f)
	
	LOG_mf2.append(pt2[0])
	NNC_mf2.append(pt2[1])
	SVC_mf2.append(pt2[2])
	KNN_mf2.append(pt2[3])
	GNB_mf2.append(pt2[4])

	if rnx == rn:
		LOG_af.append(acc1[0])
		NNC_af.append(acc1[1])
		SVC_af.append(acc1[2])
		RFC_af.append(acc1[3])
		KNN_af.append(acc1[4])
		GNB_af.append(acc1[5])
		WS_af.append(acc1[6])

		LOG_af2.append(acc2[0])
		NNC_af2.append(acc2[1])
		SVC_af2.append(acc2[2])
		KNN_af2.append(acc2[3])
		GNB_af2.append(acc2[4])

	#####################################################
	# Save the partial predictions
	#####################################################
	if rnx == rn:

		fullpred = []

		# original before merge
		ivec = list(df_test["PassengerId"])

		# after merge
		#ivec_m = list(df_test_m["pid"])
		#ivec_f = list(df_test_f["pid"])

		# save partial results
		ut.save_results(ivec, ivec_m, ivec_f, mp_m1, mp_m2, mp_f1, mp_f2, df_test_m, df_test_f)

#############################################################
# Compare models in ROC space
#############################################################

CM_m = [LOG_mm, NNC_mm, SVC_mm, RFC_mm, KNN_mm, GNB_mm, WS_mm, LOG_mm2, NNC_mm2, SVC_mm2, KNN_mm2, GNB_mm2]
ACC_m = [LOG_am, NNC_am, SVC_am, RFC_am, KNN_am, GNB_am, WS_am, LOG_am2, NNC_am2, SVC_am2, KNN_am2, GNB_am2]
CM_f = [LOG_mf, NNC_mf, SVC_mf, RFC_mf, KNN_mf, GNB_mf, WS_mf, LOG_mf2, NNC_mf2, SVC_mf2, KNN_mf2, GNB_mf2]
ACC_f = [LOG_af, NNC_af, SVC_af, RFC_af, KNN_af, GNB_af, WS_af, LOG_af2, NNC_af2, SVC_af2, KNN_af2, GNB_af2]

pvec = []
for x in range(len(CM_m)):

	rtuple = ut.ave_std(CM_m[x])
	pvec.append(rtuple)
	rtuple = ut.ave_std(CM_f[x])
	pvec.append(rtuple)

print "##############################################################"

n_hm = len(df_test_m)
n_hf = len(df_test_f)
n_em = len(df_tm) - len(df_test_m)
n_ef = len(df_tf) - len(df_test_f)
nm = len(df_tm)
nf = len(df_tf)

print
print "Number easy test men: ", n_em
print "Number easy test women: ", n_ef
print
print "Number hard test men: ", n_hm
print "Number hard test women: ", n_hf
print
print "Number test men: ", nm
print "Number test women: ", nf
print
for x in range(len(ACC_m)):
	tmp = np.asarray(ACC_m[x])
	ave_m = tmp.mean()
	print lvec[x] + ' (male): ' + str(ave_m)

	tmp = np.asarray(ACC_f[x])
	ave_f = tmp.mean()
	print lvec[x] + ' (female): ' + str(ave_f)
	print lvec[x] + ': ' + str((ave_m * nm + ave_f * nf) / float(nm + nf))
	print

print
print "##############################################################"

if ROC:

	mdpi = 96
	plt.figure(figsize=(1200/mdpi,1200/mdpi), dpi=mdpi)
	cm = plt.get_cmap('hsv')
	NUM = int(len(pvec)/2)
	cvec = [cm(i/float(NUM)) for i in range(NUM)]
	# Plot ROC
	i = 0
	for x in range(int(len(pvec)/2)):

		plt.errorbar(pvec[i][0], pvec[i][1], xerr=pvec[i][2], yerr=pvec[i][3], c=cvec[x], label=lvec[x] + " (male)", marker='.', ms=20)
		i += 1
		plt.errorbar(pvec[i][0], pvec[i][1], xerr=pvec[i][2], yerr=pvec[i][3], c=cvec[x], label=lvec[x] + " (female)", marker='d', ms=15)
		i += 1

	plt.plot([0,1], [0,1], ls='--', c='r', lw=2)
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.xlim((0,1))
	plt.ylim((0,1))
	plt.legend(loc=4)
	plt.savefig('compare_models_mf.png')
	plt.show()

########################################################################
# Save model parameters and figure
########################################################################

with open('model_param_m4.csv', 'w') as csvFile:
    	writer = csv.writer(csvFile)
    	writer.writerow(["nnc param 1: "] + m1_param_m['nnc'])
    	writer.writerow(["svc param 1: "] + m1_param_m['svc'])
    	writer.writerow(["rfc param 1: "] + m1_param_m['rfc'])
    	writer.writerow(["knn param 1: "] + m1_param_m['knn'])

    	writer.writerow(["nnc param 2: "] + m2_param_m['nnc'])
    	writer.writerow(["svc param 2: "] + m2_param_m['svc'])
    	writer.writerow(["rfc param 2: "] + m2_param_m['rfc'])
    	writer.writerow(["knn param 2: "] + m2_param_m['knn'])

	for x in range(len(ACC_m)):
		tmp = np.asarray(ACC_m[x])
		ave = tmp.mean()
		writer.writerow([lvec_acc[x], str(ave)])

with open('model_param_f4.csv', 'w') as csvFile:
    	writer = csv.writer(csvFile)
    	writer.writerow(["nnc param 1: "] + m1_param_f['nnc'])
    	writer.writerow(["svc param 1: "] + m1_param_f['svc'])
    	writer.writerow(["rfc param 1: "] + m1_param_f['rfc'])
    	writer.writerow(["knn param 1: "] + m1_param_f['knn'])

    	writer.writerow(["nnc param 2: "] + m2_param_f['nnc'])
    	writer.writerow(["svc param 2: "] + m2_param_f['svc'])
    	writer.writerow(["rfc param 2: "] + m2_param_f['rfc'])
    	writer.writerow(["knn param 2: "] + m2_param_f['knn'])

	for x in range(len(ACC_f)):
		tmp = np.asarray(ACC_f[x])
		ave = tmp.mean()
		writer.writerow([lvec_acc[x], str(ave)])

csvFile.close()

#print
#print "PCA_m: ", PCA_m
#print
#print "PCA_f: ", PCA_f
#print
#print "wd: ", wd
#print
#print "reduced_m: ", REDUCED_m
#print
#print "reduced_f: ", REDUCED_f
#print
