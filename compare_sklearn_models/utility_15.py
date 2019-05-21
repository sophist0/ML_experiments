import numpy as np
import itertools
import copy
import pandas as pd
import math
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB

import csv
from csv import reader

def write_csv_results(path, ivec, fullpred, name_m, name_f):

	# Name Format
	# predname_sex_level

	#########################################################
	# Write full prediction results as a CSV
	#########################################################

	with open(path+name_m + "_" + name_f +'.csv', 'w') as csvfile:
		writer = csv.writer(csvfile, delimiter=',')
		writer.writerow(['PassengerId','Survived'])
		for x in range(len(ivec)):
			writer.writerow([ivec[x],fullpred[x]])

	return ivec, fullpred 

def save_results(ivec, ivec_m, ivec_f, mp_m1, mp_m2, mp_f1, mp_f2, df_test_m, df_test_f):

	#########################################################
	# Saves results as a Pickle
	#########################################################
	path="predictions/"
	pickle.dump(ivec, open(path+'test_index.p','wb'))	
	pickle.dump(ivec_m, open(path+'test_index_m.p','wb'))	
	pickle.dump(ivec_f, open(path+'test_index_f.p','wb'))	

	pickle.dump(mp_m1, open(path+'pred1_m.p','wb'))	
	pickle.dump(mp_m2, open(path+'pred2_m.p','wb'))	
	pickle.dump(mp_f1, open(path+'pred1_f.p','wb'))	
	pickle.dump(mp_f2, open(path+'pred2_f.p','wb'))	

	pickle.dump(df_test_m, open(path+'df_test_m.p','wb'))	
	pickle.dump(df_test_f, open(path+'df_test_f.p','wb'))	

def class_easy_test(df, tcol_m ,tcol_f):

	df_test_m = df[df['sex']==0]
	df_test_f = df[df['sex']==1]

	################################################################
	# Separate Samples
	################################################################

	# predictions
	pvec_m = []
	pvec_f = []

	# male unpredicted
	m_uvec = []
	mdx = 0
	for x in range(len(df_test_m['sex'])):

		done = False

		# TEST
		if 1:
			# adult
			if df_test_m['Master.'].iloc[x] == 0:

				if df_test_m['cabin_unknown'].iloc[x]==1:
					pvec_m.append(0)
					done = True

		if done == False:
			m_uvec.append(mdx)
			pvec_m.append(-1)

		mdx += 1

	# female unpredicted
	f_uvec = []
	fdx = 0
	for x in range(len(df_test_f['sex'])):

		done = False

		# TEST
		if 1:

			# adult
			if df_test_f['Miss.'].iloc[x] == 0:

				if df_test_f['pclass'].iloc[x]==1 or df_test_f['pclass'].iloc[x]==2 or df_test_f['cabin_unknown'].iloc[x]==0 or df_test_f['onesur'].iloc[x]==1:

					pvec_f.append(1)
					done = True

			# child
			else:

				if df_test_f['pclass'].iloc[x]==1 or df_test_f['pclass'].iloc[x]==2:

					pvec_f.append(1)
					done = True

		if done == False:
			f_uvec.append(fdx)
			pvec_f.append(-1)

		fdx += 1

	################################################################
	# Get reduced testing sets
	################################################################
	M = False
	for x in range(len(df_test_m['sex'])):
		if x in m_uvec:		
			row = df_test_m.iloc[[x]]
			if M == False:
				df_m = pd.DataFrame(row.values, columns = row.columns)
				M = True
			else:
				df_m = df_m.append(row)

	F = False
	for x in range(len(df_test_f['sex'])):
		if x in f_uvec:
			row = df_test_f.iloc[[x]]
			if F == False:
				df_f = pd.DataFrame(row.values, columns = row.columns)
				F = True
			else:
				df_f = df_f.append(row)

	df_m = df_m[['pid','survived']+tcol_m]
	df_f = df_f[['pid','survived']+tcol_f]

	return [df_m, df_f, m_uvec, f_uvec, pvec_m, pvec_f]


def calc_acc(pvec, pidx, model_pred, svec, test_b, test_n):

	# acc of easy samples
	easy_acc = 0
	easy_num = 0
	for x in range(len(pvec)):
		if pvec[x] != -1:
			easy_num += 1

			if pvec[x] == svec[x]:
				easy_acc += 1

	if easy_num > 0:
		easy_acc = easy_acc / float(easy_num)
	else:
		easy_acc = -1

	idx = 0
	for x in range(len(pvec)):
		if x in pidx:
			pvec[x] = model_pred[idx]
			idx += 1

	c = 0
	for x in range(len(pvec)):
		if pvec[x] == svec[x]:
			c += 1

	lr_acc = c / float(len(pvec))
	lr_cm = confusion_matrix(svec,pvec)
	lr_pt = (lr_cm[0][1] / float(len(test_n)) , lr_cm[1][1] / float(len(test_b)))

	return lr_acc, lr_pt, easy_acc, pvec


def class_easy(name, REDUCED_m, REDUCED_f):

	# class easy samples
	df_train = pd.read_csv(name, sep=',')
	df_train_m = df_train[df_train['sex']==0]
	df_train_f = df_train[df_train['sex']==1]

	################################################################
	# Separate Samples
	################################################################

	# predictions
	pvec_m = []
	pvec_f = []
	# male unpredicted
	m_uvec = []
	# female unpredicted
	f_uvec = []

	for x in range(len(df_train_m['sex'])):

		done = False

		# TEST
		if 1:
			# adult
			if df_train_m['Master.'].iloc[x] == 0:

				if df_train_m['cabin_unknown'].iloc[x]==1:

					pvec_m.append(0)
					done = True

		if done == False:
			m_uvec.append(x)
			pvec_m.append(-1)

	################################################################################

	for x in range(len(df_train_f['sex'])):

		done = False

		# TEST
		if 1:
			# adult
			if df_train_f['Miss.'].iloc[x] == 0:

				if df_train_f['pclass'].iloc[x]==1 or df_train_f['pclass'].iloc[x]==2 or df_train_f['cabin_unknown'].iloc[x]==0 or df_train_f['onesur'].iloc[x]==1:

					pvec_f.append(1)
					done = True

			# child
			else:

				if df_train_f['pclass'].iloc[x]==1 or df_train_f['pclass'].iloc[x]==2:

					pvec_f.append(1)
					done = True

		if done == False:
			f_uvec.append(x)
			pvec_f.append(-1)

	################################################################
	# Get reduced training sets
	################################################################
	M = False
	for x in range(len(df_train_m['sex'])):
		if x in m_uvec and REDUCED_m == True:		
			row = df_train_m.iloc[[x]]
			if M == False:
				df_m = pd.DataFrame(row.values, columns = row.columns)
				M = True
			else:
				df_m = df_m.append(row)

		elif REDUCED_m == False:
			row = df_train_m.iloc[[x]]
			if M == False:
				df_m = pd.DataFrame(row.values, columns = row.columns)
				M = True
			else:
				df_m = df_m.append(row)

	F = False
	for x in range(len(df_train_f['sex'])):
		if x in f_uvec and REDUCED_f == True:
			row = df_train_f.iloc[[x]]
			if F == False:
				df_f = pd.DataFrame(row.values, columns = row.columns)
				F = True
			else:
				df_f = df_f.append(row)

		elif REDUCED_f == False:
			row = df_train_f.iloc[[x]]
			if F == False:
				df_f = pd.DataFrame(row.values, columns = row.columns)
				F = True
			else:
				df_f = df_f.append(row)

	#################################################################
	# Cut number of features for each model
	#################################################################

	tcol = list(df_m.columns)
	tcol.remove('survived')
	tcol.remove('pid')

	# Normalize feature columns
	n_std_m = df_m[tcol].std()
	n_std_f = df_f[tcol].std()
	val_m = n_std_m.values
	val_f = n_std_f.values

	tcol_m = []
	tcol_f = []
	for x in range(len(tcol)):	# get cols with zero std 
		if val_m[x] > 0:
			tcol_m.append(tcol[x])
		if val_f[x] > 0:
			tcol_f.append(tcol[x])

	df_m = df_m[['pid','survived']+tcol_m]
	df_f = df_f[['pid','survived']+tcol_f]

	n_ave_m = df_m[tcol_m].mean()
	n_std_m = df_m[tcol_m].std()
	print
	print n_ave_m
	print
	df_m[tcol_m] = (df_m[tcol_m]- n_ave_m) / n_std_m

	n_ave_f = df_f[tcol_f].mean()
	n_std_f = df_f[tcol_f].std()
	df_f[tcol_f] = (df_f[tcol_f]- n_ave_f) / n_std_f

	return [df_m, df_f, n_ave_m, n_ave_f, n_std_m, n_std_f, m_uvec, f_uvec, pvec_m, pvec_f]

def pca_trans(df, eigVec, cols):

	if eigVec == []:
		df_cor = np.corrcoef(df.T)
		print
		print df_cor
		print
		eigVal, eigVec = np.linalg.eig(df_cor)

		# Sort largest eigenvalue first
		eigVal, eigVec = zip(*sorted(zip(eigVal,eigVec),reverse=True))

	new_df = np.matmul(df,eigVec)
	print
	print cols
	print
	new_df.columns = cols

	return new_df, eigVec

def meta_df_fold_pca(df_train, df_test, train_p, test_p, tcol, wd, PCA):

	# wd == True -> with original data
	# wd == False -> without original data

	lr_pred_1 = train_p['lrc']
	nnc_pred_1 = train_p['nnc']
	svc_pred_1 = train_p['svc']
	rfc_pred_1 = train_p['rfc']
	knn_pred_1 = train_p['knn']

	lr_pred_2 = test_p['lrc']
	nnc_pred_2 = test_p['nnc']
	svc_pred_2 = test_p['svc']
	rfc_pred_2 = test_p['rfc']
	knn_pred_2 = test_p['knn']

	mcol = ["M"+str(x+1) for x in range(5)]
	tmp_1 = [lr_pred_1, nnc_pred_1, svc_pred_1, rfc_pred_1, knn_pred_1]
	tmp_1 = np.asarray(tmp_1)
	df_tmp_1 = pd.DataFrame(data=tmp_1.T, index=df_train.index, columns=mcol)

	# Normalize feature columns
	std_t1 = df_tmp_1[mcol].std()
	val_t1 = std_t1.values

	cols = []
	for x in range(len(mcol)):	# get cols with zero std 
		if val_t1[x] > 0.0:
			cols.append(mcol[x])

	ave_t1 = df_tmp_1[cols].mean()
	std_t1 = df_tmp_1[cols].std()
	df_tmp_1 = (df_tmp_1[cols]- ave_t1) / std_t1

	# PCA training
	if PCA:
		df_tmp_1, eigvec_t1 = pca_trans(df_tmp_1[cols], [], cols)

	tmp_2 = [lr_pred_2, nnc_pred_2, svc_pred_2, rfc_pred_2, knn_pred_2]
	tmp_2 = np.asarray(tmp_2)
	df_tmp_2 = pd.DataFrame(data=tmp_2.T, index=df_test.index, columns=mcol)

	df_tmp_2 = (df_tmp_2[cols]- ave_t1) / std_t1

	# PCA testing
	if PCA:
		df_tmp_2, eigvec_t2 = pca_trans(df_tmp_2[cols], eigvec_t1, cols)

	if wd == False:
		# no data reuse
		df_mtrain = df_tmp_1.join(df_train['survived'])
		df_mtest = df_tmp_2.join(df_test['survived'])
		mtcol = cols
	else:
		# data reuse
		df_mtrain = df_train.join(df_tmp_1)
		df_mtest = df_test.join(df_tmp_2)
		mtcol = tcol + cols

	return df_mtrain, df_mtest, mtcol

def construct_meta(df_train, df_test, test_b, test_n, train_p, test_p, tcol, wd, rn, PCA):

	pt = []
	acc = []

	#########################################################################
	# Meta Models
	#########################################################################

	print
	print "################################"

	#df_mtrain, df_mtest, mtcol = meta_df_fold(df_train, df_test, train_p, test_p, tcol, wd)
	df_mtrain, df_mtest, mtcol = meta_df_fold_pca(df_train, df_test, train_p, test_p, tcol, wd, PCA)

	m2_param = {}
	########################################################################
	# meta logistic regression
	########################################################################
	lr = LogisticRegression(solver='liblinear', max_iter=100, random_state=rn)
	lr.fit(df_mtrain[mtcol], df_mtrain['survived'])

	lr_acc = lr.score(df_mtest[mtcol],df_test['survived'])
	lr_pred = lr.predict(df_mtest[mtcol])
	lr_cm = confusion_matrix(df_mtest['survived'],lr_pred)
	lr_pt = (lr_cm[0][1] / float(len(test_n)) , lr_cm[1][1] / float(len(test_b)))
	pt.append(lr_pt)
	acc.append(lr_acc)

	########################################################################
	# meta ann
	########################################################################

	prange = [(0,5),(6,14),(2,4)]
	nnc_param = grid_opt_fold(mtcol, 'nnc', prange, copy.deepcopy(df_mtrain))
	print
	print "NNC meta params:"
	print nnc_param
	print

	m2_param['nnc'] = nnc_param
	nnc = MLPClassifier(solver='lbfgs', alpha=(0.0001*(math.pow(10,m2_param['nnc'][0]))), hidden_layer_sizes=(m2_param['nnc'][1],m2_param['nnc'][2]), random_state=rn) # good
	nnc.fit(df_mtrain[mtcol],df_mtrain['survived'])

	nnc_acc = nnc.score(df_mtest[mtcol],df_test['survived'])
	nnc_pred = nnc.predict(df_mtest[mtcol])
	nnc_cm = confusion_matrix(df_mtest['survived'],nnc_pred)	
	nnc_pt = (nnc_cm[0][1] / float(len(test_n)) , nnc_cm[1][1] / float(len(test_b)))
	pt.append(nnc_pt)
	acc.append(nnc_acc)

	########################################################################
	# meta svm
	########################################################################

	prange = [(1,10)]
	svc_param = grid_opt_fold(mtcol, 'svc', prange, copy.deepcopy(df_mtrain))
	print
	print "SVC meta params:"
	print svc_param
	print

	m2_param['svc'] = svc_param
	svc_c = SVC(gamma='scale', random_state=rn, kernel='rbf', shrinking=False, C=m2_param['svc'][0])
	svc_c.fit(df_mtrain[mtcol], df_mtrain['survived'])

	svc_acc = svc_c.score(df_mtest[mtcol], df_test['survived'])
	svc_pred = svc_c.predict(df_mtest[mtcol])
	svc_cm = confusion_matrix(df_mtest['survived'], svc_pred)
	svc_pt = (svc_cm[0][1] / float(len(test_n)) , svc_cm[1][1] / float(len(test_b)))
	pt.append(svc_pt)
	acc.append(svc_acc)

	########################################################################
	# meta rfc
	########################################################################

	prange = [(2,5)]
	rfc_param = grid_opt_fold(mtcol, 'rfc', prange, copy.deepcopy(df_mtrain))
	print
	print "RFC meta params:"
	print rfc_param
	print

	m2_param['rfc'] = rfc_param
	rfc = RandomForestClassifier(n_estimators=100, n_jobs=4, criterion='entropy', max_depth=m2_param['rfc'][0], random_state=rn)
	rfc.fit(df_mtrain[mtcol], df_mtrain['survived'])

	rfc_acc = rfc.score(df_mtest[mtcol], df_test['survived'])
	rfc_pred = rfc.predict(df_mtest[mtcol])
	rfc_cm = confusion_matrix(df_mtest['survived'], rfc_pred)
	rfc_pt = (rfc_cm[0][1] / float(len(test_n)) , rfc_cm[1][1] / float(len(test_b)))
	pt.append(rfc_pt)
	acc.append(rfc_acc)

	########################################################################
	# meta knn
	########################################################################
	prange = [(1,10),(1,3)]
	knn_param = grid_opt_fold(mtcol, 'knn', prange, copy.deepcopy(df_mtrain))
	print
	print "KNN meta params:"
	print knn_param
	print

	m2_param['knn'] = knn_param
	knn_c = KNeighborsClassifier(n_neighbors=m2_param['knn'][0], n_jobs=4, p=m2_param['knn'][1])
	knn_c.fit(df_mtrain[mtcol], df_train['survived'])

	knn_acc = knn_c.score(df_mtest[mtcol], df_test['survived'])
	knn_c_pred = knn_c.predict(df_mtest[mtcol])
	knn_cm = confusion_matrix(df_mtest['survived'], knn_c_pred)
	knn_pt = (knn_cm[0][1] / float(len(test_n)) , knn_cm[1][1] / float(len(test_b)))
	pt.append(knn_pt)
	acc.append(knn_acc)

	########################################################################
	# meta naive bayes
	########################################################################
	gnb_c = GaussianNB()
	gnb_c.fit(df_mtrain[mtcol], df_mtrain['survived'])

	gnb_acc = gnb_c.score(df_mtest[mtcol],df_test['survived'])
	gnb_pred = gnb_c.predict(df_mtest[mtcol])
	gnb_cm = confusion_matrix(df_mtest['survived'],lr_pred)
	gnb_pt = (gnb_cm[0][1] / float(len(test_n)) , gnb_cm[1][1] / float(len(test_b)))
	pt.append(gnb_pt)
	acc.append(gnb_acc)

	return pt, acc, m2_param


def get_roc_meta(df_train, df_test, test_b, test_n, m2_param, train_pm, test_pm, tcol, pidx, pvec, svec, wd, rn, PCA):

	################################################################################################################
	# df_train_f, df_test_f, test_bf, test_nf, m2_param_f, train_pf, test_pf, tcol_f, fu_test, pvec_f, wd, rnx
	################################################################################################################

	#df_mtrain, df_mtest, mtcol = meta_df_fold(df_train, df_test, train_pm, test_pm, tcol, wd)
	df_mtrain, df_mtest, mtcol = meta_df_fold_pca(df_train, df_test, train_pm, test_pm, tcol, wd, PCA)

	########################################################################

	pt = []
	acc = []
	pred = {}
	#print
	#print "##############################################################"
	#print "Logistic Classifier"
	#print "##############################################################"
	#print

	# male
	lr = LogisticRegression(solver='liblinear', max_iter=100, random_state=rn)
	lr.fit(df_mtrain[mtcol], df_mtrain['survived'])
	lr_pred = lr.predict(df_mtest[mtcol])

	p_tmp = copy.deepcopy(pvec)
	lr_acc, lr_pt, e_acc, lr_pred2 = calc_acc(p_tmp, pidx, lr_pred, svec, test_b, test_n)
	pt.append(lr_pt)
	acc.append(lr_acc)
	pred['lr'] = lr_pred2

	#print "##############################################################"
	#print "ANN Classifier"
	#print "##############################################################"
	#print

	nnc = MLPClassifier(solver='lbfgs', alpha=(0.0001*(math.pow(10,m2_param['nnc'][0]))), hidden_layer_sizes=(m2_param['nnc'][1],m2_param['nnc'][2]), random_state=rn) # good
	nnc.fit(df_mtrain[mtcol],df_train['survived'])
	nnc_pred = nnc.predict(df_mtest[mtcol])

	p_tmp = copy.deepcopy(pvec)
	nnc_acc, nnc_pt, e_acc, nnc_pred2 = calc_acc(p_tmp, pidx, nnc_pred, svec, test_b, test_n)
	pt.append(nnc_pt)
	acc.append(nnc_acc)
	pred['nnc'] = nnc_pred2

	#print "##############################################################"
	#print "SVM Classifier"
	#print "##############################################################"
	#print

	svc_c = SVC(gamma='scale', random_state=rn, kernel='rbf', shrinking=False, C=m2_param['svc'][0])
	svc_c.fit(df_mtrain[mtcol], df_train['survived'])
	svc_pred = svc_c.predict(df_mtest[mtcol])

	p_tmp = copy.deepcopy(pvec)
	svc_acc, svc_pt, e_acc, svc_pred2 = calc_acc(p_tmp, pidx, svc_pred, svec, test_b, test_n)
	pt.append(svc_pt)
	acc.append(svc_acc)
	pred['svc'] = svc_pred2

	#print "##############################################################"
	#print "Random Forest Classifier"
	#print "##############################################################"
	#print

	rfc = RandomForestClassifier(n_estimators=100, n_jobs=4, criterion='entropy', max_depth=m2_param['rfc'][0], random_state=rn)
	rfc.fit(df_mtrain[mtcol], df_train['survived'])
	rfc_pred = rfc.predict(df_mtest[mtcol])

	p_tmp = copy.deepcopy(pvec)
	rfc_acc, rfc_pt, e_acc, rfc_pred2 = calc_acc(p_tmp, pidx, rfc_pred, svec, test_b, test_n)
	pt.append(rfc_pt)
	acc.append(rfc_acc)
	pred['rfc'] = rfc_pred2

	#print "##############################################################"
	#print "KNN Classifier"
	#print "##############################################################"
	#print

	knn_c = KNeighborsClassifier(n_neighbors=m2_param['knn'][0], n_jobs=4, p=m2_param['knn'][1])
	knn_c.fit(df_mtrain[mtcol], df_train['survived'])
	knn_c_pred = knn_c.predict(df_mtest[mtcol])

	p_tmp = copy.deepcopy(pvec)
	knn_acc, knn_pt, e_acc, knn_pred2 = calc_acc(p_tmp, pidx, knn_c_pred, svec, test_b, test_n)
	pt.append(knn_pt)
	acc.append(knn_acc)
	pred['knn'] = knn_pred2

	#print "##############################################################"
	#print "GNB Classifier"
	#print "##############################################################"
	#print

	gnb_c = GaussianNB()
	gnb_c.fit(df_mtrain[mtcol], df_train['survived'])
	gnb_c_pred = gnb_c.predict(df_mtest[mtcol])

	p_tmp = copy.deepcopy(pvec)
	gnb_acc, gnb_pt, e_acc, gnb_pred2 = calc_acc(p_tmp, pidx, gnb_c_pred, svec, test_b, test_n)
	pt.append(gnb_pt)
	acc.append(gnb_acc)
	pred['gnb'] = gnb_pred2

	return pt, acc, pred


def get_roc(df_train, df_test, test_b, test_n, m1_param, pidx, pvec, svec, men, rn):

	pt = []
	acc = []
	pred ={}
	#print
	#print "##############################################################"
	#print "Logistic Classifier"
	#print "##############################################################"
	#print

	# male
	tcol = m1_param['lr_f']
	lr = LogisticRegression(solver='liblinear', max_iter=100, random_state=rn)
	lr.fit(df_train[tcol], df_train['survived'])
	lr_pred = lr.predict(df_test[tcol])

	p_tmp = copy.deepcopy(pvec)
	lr_acc, lr_pt, e_acc, lr_pred2 = calc_acc(p_tmp, pidx, lr_pred, svec, test_b, test_n)

	pt.append(lr_pt)
	acc.append(lr_acc)
	pred['lr'] = lr_pred2
	#print "##############################################################"
	#print "ANN Classifier"
	#print "##############################################################"
	#print

	tcol = m1_param['nnc_f']
	nnc = MLPClassifier(solver='lbfgs', alpha=(0.0001*(math.pow(10,m1_param['nnc'][0]))), hidden_layer_sizes=(m1_param['nnc'][1],m1_param['nnc'][2]), random_state=rn) # good
	nnc.fit(df_train[tcol],df_train['survived'])
	nnc_pred = nnc.predict(df_test[tcol])

	p_tmp = copy.deepcopy(pvec)
	nnc_acc, nnc_pt, e_acc, nnc_pred2 = calc_acc(p_tmp, pidx, nnc_pred, svec, test_b, test_n)
	pt.append(nnc_pt)
	acc.append(nnc_acc)
	pred['nnc'] = nnc_pred2
	#print "##############################################################"
	#print "SVM Classifier"
	#print "##############################################################"
	#print

	tcol = m1_param['svc_f']
	svc_c = SVC(gamma='scale', random_state=rn, kernel='rbf', shrinking=False, C=m1_param['svc'][0])
	svc_c.fit(df_train[tcol], df_train['survived'])
	svc_pred = svc_c.predict(df_test[tcol])

	p_tmp = copy.deepcopy(pvec)
	svc_acc, svc_pt, e_acc, svc_pred2 = calc_acc(p_tmp, pidx, svc_pred, svec, test_b, test_n)
	pt.append(svc_pt)
	acc.append(svc_acc)
	pred['svc'] = svc_pred2
	#print "##############################################################"
	#print "Random Forest Classifier"
	#print "##############################################################"
	#print

	tcol = m1_param['rfc_f']
	rfc = RandomForestClassifier(n_estimators=100, n_jobs=4, criterion='entropy', max_depth=m1_param['rfc'][0], random_state=rn)
	rfc.fit(df_train[tcol], df_train['survived'])
	rfc_pred = rfc.predict(df_test[tcol])

	p_tmp = copy.deepcopy(pvec)
	rfc_acc, rfc_pt, e_acc, rfc_pred2 = calc_acc(p_tmp, pidx, rfc_pred, svec, test_b, test_n)
	pt.append(rfc_pt)
	acc.append(rfc_acc)
	pred['rfc'] = rfc_pred2
	#print "##############################################################"
	#print "KNN Classifier"
	#print "##############################################################"
	#print

	tcol = m1_param['knn_f']
	knn_c = KNeighborsClassifier(n_neighbors=m1_param['knn'][0], n_jobs=4, p=m1_param['knn'][1])
	knn_c.fit(df_train[tcol], df_train['survived'])
	knn_c_pred = knn_c.predict(df_test[tcol])

	p_tmp = copy.deepcopy(pvec)
	knn_acc, knn_pt, e_acc, knn_pred2 = calc_acc(p_tmp, pidx, knn_c_pred, svec, test_b, test_n)
	pt.append(knn_pt)
	acc.append(knn_acc)
	pred['knn'] = knn_pred2

	#print "##############################################################"
	#print "Naive Bayes Classifier"
	#print "##############################################################"
	#print

	tcol = m1_param['gnb_f']
	gnb_c = GaussianNB()
	gnb_c.fit(df_train[tcol], df_train['survived'])
	gnb_c_pred = gnb_c.predict(df_test[tcol])

	p_tmp = copy.deepcopy(pvec)
	gnb_acc, gnb_pt, e_acc, gnb_pred2 = calc_acc(p_tmp, pidx, gnb_c_pred, svec, test_b, test_n)
	pt.append(gnb_pt)
	acc.append(gnb_acc)
	pred['gnb'] = gnb_pred2
	#print "##############################################################"
	#print "Women Survive Classifier"
	#print "##############################################################"
	#print

	if men == True:
		ws_pred = np.zeros(len(df_test))
	else:
		ws_pred = np.ones(len(df_test))

	p_tmp = copy.deepcopy(pvec)
	ws_acc, ws_pt, e_acc, ws_pred2 = calc_acc(p_tmp, pidx, ws_pred, svec, test_b, test_n)
	pt.append(ws_pt)
	acc.append(ws_acc)
	pred['ws'] = ws_pred2

	print
	print "############################"
	print "MEN: ", men
	print "easy: ", e_acc
	print
	print "lr: ", lr_acc
	print "nnc: ", nnc_acc
	print "svc: ", svc_acc
	print "nnc: ", nnc_acc
	print "knn: ", knn_acc
	print "gnb: ", gnb_acc
	print "ws_acc: ", ws_acc
	print

	return pt, acc, e_acc, pred

##############################################################

def zero_neg(np_vec):

	ovec = np.ones(len(np_vec))
	np_vec += -1*(np_vec < ovec)

	return np_vec

def ave_std(l):

	xvec, yvec = zip(*l)
	
	xvec = np.asarray(xvec)
	yvec = np.asarray(yvec)
	xave = xvec.mean()
	xstd = xvec.std()
	yave = yvec.mean()
	ystd = yvec.std()

	return (xave,yave,xstd,ystd)

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

	print
	print PMAT
	print
	pcombo = list(itertools.product(*PMAT))
	print "Number of parameter combos: ", len(pcombo)
	print
	kf = KFold(n_splits=5, random_state=rn, shuffle=True)
	bparam = [0 for x in range(len(pvec))]
	bacc = 0
	for x in range(len(pcombo)):

		avec = []
		if model == 'nnc':
			print
			print "NNC Combos tested: " + str(x) + " of " + str(len(pcombo))
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
			print
			print "SVC Combos tested: " + str(x) + " of " + str(len(pcombo))
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
			print
			print "RFC Combos tested: " + str(x) + " of " + str(len(pcombo))
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
			print
			print "KNN Combos tested: " + str(x) + " of " + str(len(pcombo))
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

	return bparam


def fam_features(fname):

	###############################################################
	# construct family features from the training features
	###############################################################

	f = open(fname,"r")
	fdict={}
	pvec = []
	i = 0
	for l in reader(f):
		if i > 0:
			key1 = l[8] # ticket!!!!
			if key1 in fdict:
				# (pid, survive)
				fdict[key1].append((l[0],l[1]))
			else:
				fdict[key1] = [(l[0],l[1])]
			# (ticket,pid)
			pvec.append((l[8],l[0]))
		i += 1
	f.close()

	# At least one family member in the training set dies.
	f1 = {}
	# At least one family member in the training set survives.
	f2 = {}
	for el in pvec:

		flist = fdict[el[0]]

		if len(flist) > 1:
			f1[el[1]] = 0
			f2[el[1]] = 0
			for x in range(len(flist)):

				# onedie
				if flist[x][0] != el[1] and int(flist[x][1]) == 0:
					f1[el[1]] = 1
				# onesur
				if flist[x][0] != el[1] and int(flist[x][1]) == 1:
					f2[el[1]] = 1
	return f1, f2, fdict


def train_transform(load_f, save_f):

	# unordered dictionaries
	onedie, onesur, tickets = fam_features(load_f)

	f = open(load_f,"r")
	#################################
	# features
	#################################
	# passenger id
	pvec = []
	# survived
	uvec = []
	# class
	lvec = []
	# surnames
	mat = [[] for x in range(4)]
	# gender
	svec = []
	# age
	cvec = []
	# age unknown
	kvec = []
	# sibling or spouse
	avec = []
	# parent
	bvec = []

	######################
	# Have or not
	######################
	# ticket
	tvec = []
	# fare
	fvec = []
	# cabin
	ivec = []

	######################
	# Family features
	######################
	f1 = []
	f2 = []
	f3 = []
	i = 0
	for l in reader(f):

		if i > 0:
			pvec.append(int(float(l[0])))
			uvec.append(int(float(l[1])))
			lvec.append(int(float(l[2])))

			# surname [Mr. 0, Mrs. 1, Miss. 2, Master. 3, Don.(ELSE) 4,]
			name_tmp = l[3].split(" ")	
			if "Mr." in name_tmp:
				mat[0].append(1)
				mat[1].append(0)
				mat[2].append(0)
				mat[3].append(0)
			elif "Mrs." in name_tmp:
				mat[0].append(0)
				mat[1].append(1)
				mat[2].append(0)
				mat[3].append(0)
			elif "Miss." in name_tmp:
				mat[0].append(0)
				mat[1].append(0)
				mat[2].append(1)
				mat[3].append(0)
			elif "Master." in name_tmp:
				mat[0].append(0)
				mat[1].append(0)
				mat[2].append(0)
				mat[3].append(1)
			else:
				mat[0].append(0)
				mat[1].append(0)
				mat[2].append(0)
				mat[3].append(0)
			# gender
			if l[4] == "female":
				svec.append(1)
			else:
				svec.append(0)
			# age
			if l[5] != '':
				cvec.append(int(float(l[5])))
			else: 
				# default 18 years old
				cvec.append(18)
			
			if l[5] == '':
				# age unknown
				kvec.append(1)
			else:
				kvec.append(0)
			# silsp
			if int(float(l[6])) > 0:
				avec.append(1)
			else:
				avec.append(0)
			# parch
			if int(float(l[7])) > 0:
				bvec.append(1)
			else:
				bvec.append(0)
			# ticket
			if l[8] == '':
				tvec.append(1)
			else:
				tvec.append(0)
			# fare
			if l[9] == '' or int(float(l[9])) == 0:
				fvec.append(1)
			else:
				fvec.append(0)
			# cabin
			if l[10] == '':
				ivec.append(1)
			else:
				ivec.append(0)

			# at least one family member dies
			if l[0] in onedie:
				f1.append(int(onedie[l[0]]))
			else:
				f1.append(0)

			# at least one family member survives
			if l[0] in onesur:
				f2.append(int(onesur[l[0]]))
			else:
				f2.append(0)

			# has family members
			if l[8] in tickets and (len(tickets[l[8]]) > 1):
				f3.append(1)
			else:
				f3.append(0)

		else:
			pass

		i += 1

	f.close()

	# write the features to a CSV file and a Numpy array
	train_data = []

	f = open(save_f,"w")

	#################################################################
	# All one in ticket existing causes crash
	#################################################################

        f.write('pid,survived,pclass,Mr.,Mrs.,Miss.,Master.,sex,age,age_unknown,sibling/spouse,parent,fare_unknown,cabin_unknown,onedie,onesur,family')
        f.write('\n')
        for x in range(len(uvec)):
                tmp = [pvec[x],uvec[x], lvec[x], mat[0][x], mat[1][x],mat[2][x], mat[3][x], svec[x], cvec[x], kvec[x], avec[x], bvec[x], fvec[x], ivec[x], f1[x], f2[x], f3[x]]

                tmp_str =  numvec_to_strvec(copy.deepcopy(tmp))
                new_line = ','.join(tmp_str)
                f.write(new_line)
                f.write('\n')
                train_data.append(tmp)

        f.close()

	return onedie, onesur, tickets

def test_transform(load_f, save_f, onedie, onesur, tickets):

	f = open(load_f,"r")
	#################################
	# features
	#################################
	# passenger id
	pvec = []
	# survived
	uvec = []
	# class
	lvec = []
	# surnames
	mat = [[] for x in range(4)]
	# gender
	svec = []
	# age
	cvec = []
	# age unknown
	kvec = []
	# sibling or spouse
	avec = []
	# parent
	bvec = []

	######################
	# Have or not
	######################
	# ticket
	tvec = []
	# fare
	fvec = []
	# cabin
	ivec = []

	######################
	# Family features
	######################
	f1 = []
	f2 = []
	f3 = []
	i = 0
	for l in reader(f):

		if i > 0:
			
			pvec.append(int(float(l[0])))
			uvec.append(int(float(l[1])))
			lvec.append(int(float(l[2])))

			# surname [Mr. 0, Mrs. 1, Miss. 2, Master. 3, Don.(ELSE) 4,]
			name_tmp = l[3].split(" ")	
			if "Mr." in name_tmp:
				mat[0].append(1)
				mat[1].append(0)
				mat[2].append(0)
				mat[3].append(0)
			elif "Mrs." in name_tmp:
				mat[0].append(0)
				mat[1].append(1)
				mat[2].append(0)
				mat[3].append(0)
			elif "Miss." in name_tmp:
				mat[0].append(0)
				mat[1].append(0)
				mat[2].append(1)
				mat[3].append(0)
			elif "Master." in name_tmp:
				mat[0].append(0)
				mat[1].append(0)
				mat[2].append(0)
				mat[3].append(1)
			else:
				mat[0].append(0)
				mat[1].append(0)
				mat[2].append(0)
				mat[3].append(0)
			# gender
			if l[4] == "female":
				svec.append(1)
			else:
				svec.append(0)
			# age
			if l[5] != '':
				cvec.append(int(float(l[5])))
			else: 
				# default 18 years old
				cvec.append(18)
			
			if l[5] == '':
				# age unknown
				kvec.append(1)
			else:
				kvec.append(0)
			# silsp
			if int(float(l[6])) > 0:
				avec.append(1)
			else:
				avec.append(0)
			# parch
			if int(float(l[7])) > 0:
				bvec.append(1)
			else:
				bvec.append(0)
			# ticket
			if l[8] == '':
				tvec.append(1)
			else:
				tvec.append(0)
			# fare
			if l[9] == '' or int(float(l[9])) == 0:
				fvec.append(1)
			else:
				fvec.append(0)
			# cabin
			if l[10] == '':
				ivec.append(1)
			else:
				ivec.append(0)

			# at least one family member dies
			if l[0] in onedie:
				f1.append(int(onedie[l[0]]))
			else:
				f1.append(0)

			# at least one family member survives
			if l[0] in onesur:
				f2.append(int(onesur[l[0]]))
			else:
				f2.append(0)

			# has family members
			if l[8] in tickets and (len(tickets[l[8]]) > 1):
				f3.append(1)
			else:
				f3.append(0)

		else:
			pass

		i += 1

	f.close()

	# write the features to a CSV file and a Numpy array
	train_data = []

	f = open(save_f,"w")

	#################################################################
	# All one in ticket existing causes crash
	#################################################################

        f.write('pid,survived,pclass,Mr.,Mrs.,Miss.,Master.,sex,age,age_unknown,sibling/spouse,parent,fare_unknown,cabin_unknown,onedie,onesur,family')
        f.write('\n')
        for x in range(len(uvec)):
                tmp = [pvec[x],uvec[x], lvec[x], mat[0][x], mat[1][x],mat[2][x], mat[3][x], svec[x], cvec[x], kvec[x], avec[x], bvec[x], fvec[x], ivec[x], f1[x], f2[x], f3[x]]

                tmp_str =  numvec_to_strvec(copy.deepcopy(tmp))
                new_line = ','.join(tmp_str)
                f.write(new_line)
                f.write('\n')
                train_data.append(tmp)

        f.close()


def numvec_to_strvec(vec):

	for x in range(len(vec)):
		vec[x] = str(vec[x])

	return vec	
