#!/usr/bin/env python

##########################################################################
# Lists features in the order which they contribute the most accuracy
# to the model greedily.
#
# Notice the accuracy is not monotonically increasing in the number of
# features added.
##########################################################################

import pandas as pd
import itertools
import copy
import numpy as np
import pickle
import math
import utility_15 as ut

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB

##########################################################################

def main():

	###############################
	# Data parameters

	PCA = False
	# both true or false distinction turned out not to be useful here
	REDUCED_m = True
	REDUCED_f = True
	###############################

	###############################
	# Select model to test

	#pvec = []
	#model = 'lr'

	#pvec = [(0,5),(6,14),(2,4)]
	#model = 'nnc'

	pvec =[(1,10)]
	model = 'svc'
	
	#pvec = [(2,5)]
	#model = 'rfc'

	#pvec = [(1,10),(1,3)]
	#model = 'knn'

	#pvec = []
	#model = 'gnb'
	###############################

	if PCA and REDUCED_m and REDUCED_f:
		path="fsets_pca_red/"
	elif not PCA and REDUCED_m and REDUCED_f:
		path="fsets_red/"
	elif PCA and ((not REDUCED_m) and (not REDUCED_f)):
		path="fsets_pca/"
	elif not PCA and ((not REDUCED_m) and (not REDUCED_f)):
		path="fsets/"

	save_f = "data/train_3.csv"
 	[df_train_m, df_train_f, n_ave_m, n_ave_f, n_std_m, n_std_f, mu_train, fu_train, ptm, ptf] = ut.class_easy(save_f,REDUCED_m,REDUCED_f)

	tcol_m = list(df_train_m.columns)
	tcol_f = list(df_train_f.columns)

	tcol_mo = tcol_m[2:len(tcol_m)]
	tcol_fo = tcol_f[2:len(tcol_f)]
	print
	print tcol_m
	print tcol_f
	print

	tcol_m = []
	tcol_f = []
	for x in range(len(tcol_mo)):
		c_std_m = df_train_m[tcol_mo[x]].std()
		if c_std_m > 0:
			tcol_m.append(tcol_mo[x])

	for x in range(len(tcol_fo)):
		c_std_f = df_train_f[tcol_fo[x]].std()
		if c_std_f > 0:
			tcol_f.append(tcol_fo[x])

	# remember to normalize
	ave_m = df_train_m[tcol_m].mean()
	std_m = df_train_m[tcol_m].std()
	ave_f = df_train_f[tcol_f].mean()
	std_f = df_train_f[tcol_f].std()

	df_train_m[tcol_m] = (df_train_m[tcol_m] - ave_m) / std_m
	df_train_f[tcol_f] = (df_train_f[tcol_f] - ave_f) / std_f

	############################################################
	# PCA
	############################################################
	if PCA:
		# training set
		cm = ["C"+str(x+1) for x in range(len(tcol_m))]
		tmp_m1 = df_train_m['survived'].to_frame()
		tmp_m2, eigvec_m = ut.pca_trans(df_train_m[tcol_m], [], cm)
		tcol_m = cm

		cf = ["C"+str(x+1) for x in range(len(tcol_f))]
		tmp_f1 = df_train_f['survived'].to_frame()
		tmp_f2, eigvec_f = ut.pca_trans(df_train_f[tcol_f], [], cf)
		tcol_f = cf

		df_train_m = tmp_m1.join(tmp_m2, how='left')
		df_train_f = tmp_f1.join(tmp_f2, how='left')

	print
	print "####################################################################"
	print "Male"
	print "####################################################################"

	fset_m, avec_m, bvec_m = greedy_features(df_train_m,tcol_m,model,pvec)
	print
	print "fset: ", fset_m
	print
	print "avec: ", avec_m
	print
	print "bvec: ", bvec_m
	print

	fdict = {}
	fdict['features'] = fset_m
	fdict['accuracy'] = avec_m
	fdict['parameters'] = bvec_m

	pickle.dump(fdict, open(path+model+'_m.p','wb'))

	##########################################################################

	print
	print "#####################################################################"
	print "Female"
	print "#####################################################################"

	fset_f, avec_f, bvec_f = greedy_features(df_train_f,tcol_f,model,pvec)
	print
	print "fset: ", fset_f
	print
	print "avec: ", avec_f
	print
	print "bvec: ", bvec_f
	print

	fdict = {}
	fdict['features'] = fset_f
	fdict['accuracy'] = avec_f
	fdict['parameters'] = bvec_f

	pickle.dump(fdict, open(path+model+'_f.p','wb'))


def greedy_features(df_train,tcol1,model,pvec):

	tcol2 = copy.deepcopy(tcol1)
	fset = []
	avec = []
	bvec = []
	while tcol2 != []:
		print
		print model + " number of features selected " + str(len(fset)) + " of " + str(len(tcol1))
		bacc = 0
		bf = -1
		bparam = []
		for y in range(len(tcol2)):
			tset = fset + [tcol2[y]]
			param, acc = grid_opt_fold(tset,model,pvec,df_train)
			if acc >= bacc:
				bf = y
				bacc = acc
				bparam = param		

		bvec.append(bparam)
		avec.append(bacc)
		fset = fset + [tcol2[bf]]
		tcol2.remove(tcol2[bf])

	return fset, avec, bvec

def grid_opt_fold(tcol, model, pvec, df_train):

	#########################################################
	# 5-FOLD
	#########################################################

	# model -> gives model to test
	# pvec is the parameters with the tuples (min, max)
	# ntrial -> number of trials

	rn = 1	
	PMAT = []
	for x in range(len(pvec)):
		PMAT.append([x for x in range(pvec[x][0],pvec[x][1]+1)])

	pcombo = list(itertools.product(*PMAT))
	kf = KFold(n_splits=5, random_state=rn, shuffle=True)
	bparam = [0 for x in range(len(pvec))]
	bacc = 0
	for x in range(len(pcombo)):

		avec = []
		if model == 'lr':
			lr = LogisticRegression(solver='liblinear', max_iter=100, random_state=rn)
			for train_idx, val_idx in kf.split(df_train):
				lr.fit(df_train[tcol].values[train_idx],df_train['survived'].values[train_idx])
				acc = lr.score(df_train[tcol].values[val_idx],df_train['survived'].values[val_idx])
				avec.append(acc)
				
			am = sum(avec) / float(len(avec))
			if am > bacc:
				bacc = am

		if model == 'nnc':
			nnc = MLPClassifier(solver='lbfgs', alpha=(0.0001*(math.pow(10,pcombo[x][0]))), hidden_layer_sizes=(pcombo[x][1],pcombo[x][2]), random_state=rn) 
			for train_idx, val_idx in kf.split(df_train):
				nnc.fit(df_train[tcol].values[train_idx],df_train['survived'].values[train_idx])
				acc = nnc.score(df_train[tcol].values[val_idx],df_train['survived'].values[val_idx])
				avec.append(acc)
				
			am = sum(avec) / float(len(avec))
			if am > bacc:
				bacc = am
				for z in range(len(pcombo[x])):
					bparam[z] = pcombo[x][z]

		if model == 'svc':
          		svc_c = SVC(gamma='scale', kernel='rbf', shrinking=False, C=pcombo[x][0], random_state=rn)
			for train_idx, val_idx in kf.split(df_train):
				svc_c.fit(df_train[tcol].values[train_idx],df_train['survived'].values[train_idx])
				acc = svc_c.score(df_train[tcol].values[val_idx],df_train['survived'].values[val_idx])
				avec.append(acc)
				
			am = sum(avec) / float(len(avec))
			if am > bacc:
				bacc = am
				for z in range(len(pcombo[x])):
					bparam[z] = pcombo[x][z]

		if model == 'rfc':
			rfc = RandomForestClassifier(n_estimators=100, n_jobs=4, criterion='entropy', max_depth=pcombo[x][0], random_state=rn)
			for train_idx, val_idx in kf.split(df_train):
				rfc.fit(df_train[tcol].values[train_idx],df_train['survived'].values[train_idx])
				acc = rfc.score(df_train[tcol].values[val_idx],df_train['survived'].values[val_idx])
				avec.append(acc)
				
			am = sum(avec) / float(len(avec))
			if am > bacc:
				bacc = am
				for z in range(len(pcombo[x])):
					bparam[z] = pcombo[x][z]

		if model == 'knn':
          		knn_c = KNeighborsClassifier(n_neighbors=pcombo[x][0], n_jobs=4, p=pcombo[x][1])
			for train_idx, val_idx in kf.split(df_train):
				knn_c.fit(df_train[tcol].values[train_idx],df_train['survived'].values[train_idx])
				acc = knn_c.score(df_train[tcol].values[val_idx],df_train['survived'].values[val_idx])
				avec.append(acc)
				
			am = sum(avec) / float(len(avec))
			if am > bacc:
				bacc = am
				for z in range(len(pcombo[x])):
					bparam[z] = pcombo[x][z]

		if model == 'gnb':
			gnb = GaussianNB()
			for train_idx, val_idx in kf.split(df_train):
				gnb.fit(df_train[tcol].values[train_idx],df_train['survived'].values[train_idx])
				acc = gnb.score(df_train[tcol].values[val_idx],df_train['survived'].values[val_idx])
				avec.append(acc)
				
			am = sum(avec) / float(len(avec))
			if am > bacc:
				bacc = am

	return bparam, bacc

if __name__ == '__main__':
	main()
