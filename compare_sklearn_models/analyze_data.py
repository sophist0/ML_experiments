#!/usr/bin/env python

####################################################################
# Look at how the data and its features interact with the labels.
####################################################################

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import utility_15 as ut

def print_surdie(df,corr,c_thr,n_thr):

	# df - dataframe
	# corr - correlation dataframe
	# c_thr - correlation threshold
	# n_thr - number threshold

	t_corr = list(corr['survived'])
	t_col = list(corr.columns)
	print
	print t_col
	print
	print t_corr
	print

	# remove PID and survived from t_col
	t_col = t_col[2:len(t_col)]
	t_corr = t_corr[2:len(t_corr)]

	for x in range(len(t_col)):

		vals = list(set(df[t_col[x]]))
		ivec = []
		nvec = []
		for y in range(len(vals)):
			df_tmp = df[df[t_col[x]]==vals[y]]
			nvec.append(len(df_tmp[t_col[x]]))
			ivec.append(y)
		if abs(t_corr[x]) > c_thr and max(nvec) > n_thr:
			print "######################################"
			print
			print t_col[x]
			print
			for y in range(len(ivec)):
				df_tmp = df[df[t_col[x]]==vals[ivec[y]]]
				df_tmp_s = df_tmp[df_tmp['survived'] == max(df_tmp['survived'])]
				fs = len(df_tmp_s['survived']) / float(len(df_tmp['survived']))

				print "value: ", vals[ivec[y]]
				print "number: ", nvec[ivec[y]]
				print "survive: ", fs
				print "die: ", 1-fs
				print

	print "######################################"
	print

#################################################################################################
# Assumes target variable is in location 0
c_thr = 0.1
#n_thr = 25
n_thr = 50
#################################################################################################

##################################################
# Do we need ptm or ptf? From training data
##################################################
REDUCED_m = True
REDUCED_f = True
save_f = "data/train_3.csv"
[df_train_m, df_train_f, n_ave_m, n_ave_f, n_std_m, n_std_f, mu_train, fu_train, ptm, ptf] = ut.class_easy(save_f,REDUCED_m,REDUCED_f)

#df_std_m = df_train_m.std()
#df_ave_m = df_train_m.mean()
#df_std_f = df_train_f.std()
#df_ave_f = df_train_f.mean()
#df_train_m = (df_train_m - df_ave_m) / df_std_m

corr = df_train_m.corr(method='pearson')

print
print "-------------------------------------------------------"
print " Training Set (MALE), number: ", len(df_train_m['survived'])
print "-------------------------------------------------------"
print
print_surdie(df_train_m,corr,c_thr,n_thr)

sns.heatmap(corr, annot=True, xticklabels=corr.columns, yticklabels=corr.columns)
plt.title("Training Dataset Feature Correlation (MALE)")
plt.show()

corr = df_train_f.corr(method='pearson')

print
print "-------------------------------------------------------"
print " Training Set (FEMALE), number: ", len(df_train_f['survived'])
print "-------------------------------------------------------"
print
print_surdie(df_train_f,corr,c_thr,n_thr)

sns.heatmap(corr, annot=True, xticklabels=corr.columns, yticklabels=corr.columns)
plt.title("Training Dataset Feature Correlation (FEMALE)")
plt.show()

#################################################################################################

df_train_ma = df_train_m[df_train_m['Master.']==min(df_train_m['Master.'])]
df_train_mc = df_train_m[df_train_m['Master.']==max(df_train_m['Master.'])]

df_std_ma = df_train_ma.std()
df_ave_ma = df_train_ma.mean()
df_std_mc = df_train_mc.std()
df_ave_mc = df_train_mc.mean()

df_train_ma = (df_train_ma - df_ave_ma) / df_std_ma
corr = df_train_ma.corr(method='pearson')

print
print "-------------------------------------------------------"
print " Training Set (MALE, ADULT), number: ", len(df_train_ma['survived'])
print "-------------------------------------------------------"
print
print_surdie(df_train_ma,corr,c_thr,n_thr)

sns.heatmap(corr, annot=True, xticklabels=corr.columns, yticklabels=corr.columns)
plt.title("Training Dataset Feature Correlation (MALE, ADULT)")
plt.show()

print
print "-------------------------------------------------------"
print " Training Set (MALE, CHILD), number: ", len(df_train_mc['survived'])
print "-------------------------------------------------------"
print
print_surdie(df_train_mc,corr,c_thr,n_thr)

df_train_mc = (df_train_mc - df_ave_mc) / df_std_mc
corr = df_train_mc.corr(method='pearson')

sns.heatmap(corr, annot=True, xticklabels=corr.columns, yticklabels=corr.columns)
plt.title("Training Dataset Feature Correlation (MALE, CHILD)")
plt.show()

#################################################################################################

df_train_fa = df_train_f[df_train_f['Miss.']==min(df_train_f['Miss.'])]
df_train_fc = df_train_f[df_train_f['Miss.']==max(df_train_f['Miss.'])]

df_std_fa = df_train_fa.std()
df_ave_fa = df_train_fa.mean()
df_std_fc = df_train_fc.std()
df_ave_fc = df_train_fc.mean()

df_train_fa = (df_train_fa - df_ave_fa) / df_std_fa
corr = df_train_fa.corr(method='pearson')

print
print "-------------------------------------------------------"
print " Training Set (FEMALE, ADULT), number: ", len(df_train_fa['survived'])
print "-------------------------------------------------------"
print
print_surdie(df_train_fa,corr,c_thr,n_thr)

sns.heatmap(corr, annot=True, xticklabels=corr.columns, yticklabels=corr.columns)
plt.title("Training Dataset Feature Correlation (FEMALE, ADULT)")
plt.show()

df_train_fc = (df_train_fc - df_ave_fc) / df_std_fc
corr = df_train_fc.corr(method='pearson')

print
print "-------------------------------------------------------"
print " Training Set (FEMALE, CHILD), number: ", len(df_train_fc['survived'])
print "-------------------------------------------------------"
print
print_surdie(df_train_fc,corr,c_thr,n_thr)


sns.heatmap(corr, annot=True, xticklabels=corr.columns, yticklabels=corr.columns)
plt.title("Training Dataset Feature Correlation (FEMALE, CHILD)")
plt.show()

