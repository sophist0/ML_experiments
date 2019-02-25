import numpy as np
import itertools
import copy

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

def ave_std(l):

	xvec, yvec = zip(*l)
	
	xvec = np.asarray(xvec)
	yvec = np.asarray(yvec)
	xave = xvec.mean()
	xstd = xvec.std()
	yave = yvec.mean()
	ystd = yvec.std()

	return (xave,yave,xstd,ystd)

def grid_opt(tcol, model,pvec,ntrials, df_yes, df_no, tn):

	# model -> gives model to test
	# pvec is the parameters with the tuples (min, max)
	# ntrial -> number of trials
	
	PMAT = []
	for x in range(len(pvec)):
		PMAT.append([x for x in range(pvec[x][0],pvec[x][1]+1)])

	print
	print PMAT
	print
	pcombo = list(itertools.product(*PMAT))
	print "Number of parameter combos: ", len(pcombo)
	print

	bparam = [0 for x in range(len(pvec))]
	bacc = 0
	for x in range(len(pcombo)):

		if model == 'nnc':

			print
			print "NNC Combos tested: " + str(x) + " of " + str(len(pcombo))

			for y in range(ntrials):

         			train_b, test_b = train_test_split(df_yes, test_size=tn)
         			train_n, test_n = train_test_split(df_no, test_size=int(len(df_no)*tn/float(len(df_yes))), train_size=(len(df_yes)-tn))

         			df_test = test_b.append(test_n)
         			df_train = train_b.append(train_n)

				nnc = MLPClassifier(solver='lbfgs', alpha=pcombo[x][0], hidden_layer_sizes=(pcombo[x][1],pcombo[x][2])) # good
				nnc.fit(df_train[tcol],df_train['survived'])
				acc = nnc.score(df_test[tcol],df_test['survived'])

				if acc > bacc:
					bacc = acc
					for z in range(len(pcombo[x])):
						bparam[z] = pcombo[x][z]

		if model == 'svc':

			print
			print "SVC Combos tested: " + str(x) + " of " + str(len(pcombo))

			for y in range(ntrials):

         			train_b, test_b = train_test_split(df_yes, test_size=tn)
         			train_n, test_n = train_test_split(df_no, test_size=int(len(df_no)*tn/float(len(df_yes))), train_size=(len(df_yes)-tn))

         			df_test = test_b.append(test_n)
         			df_train = train_b.append(train_n)

          			svc_c = SVC(gamma='scale', kernel='rbf', shrinking=False, C=pcombo[x][0])
				svc_c.fit(df_train[tcol],df_train['survived'])
				acc = svc_c.score(df_test[tcol],df_test['survived'])

				if acc > bacc:
					bacc = acc
					for z in range(len(pcombo[x])):
						bparam[z] = pcombo[x][z]


		if model == 'rfc':

			print
			print "RFC Combos tested: " + str(x) + " of " + str(len(pcombo))

			for y in range(ntrials):

         			train_b, test_b = train_test_split(df_yes, test_size=tn)
         			train_n, test_n = train_test_split(df_no, test_size=int(len(df_no)*tn/float(len(df_yes))), train_size=(len(df_yes)-tn))

         			df_test = test_b.append(test_n)
         			df_train = train_b.append(train_n)

				rfc = RandomForestClassifier(n_estimators=100, n_jobs=4, criterion='entropy', max_features=pcombo[x][0], max_depth=pcombo[x][1])
				rfc.fit(df_train[tcol],df_train['survived'])
				acc = rfc.score(df_test[tcol],df_test['survived'])

				if acc > bacc:
					bacc = acc
					for z in range(len(pcombo[x])):
						bparam[z] = pcombo[x][z]


		if model == 'knn':

			print
			print "KNN Combos tested: " + str(x) + " of " + str(len(pcombo))

			for y in range(ntrials):

         			train_b, test_b = train_test_split(df_yes, test_size=tn)
         			train_n, test_n = train_test_split(df_no, test_size=int(len(df_no)*tn/float(len(df_yes))), train_size=(len(df_yes)-tn))

         			df_test = test_b.append(test_n)
         			df_train = train_b.append(train_n)

         			knn_c = KNeighborsClassifier(n_neighbors=pcombo[x][0], n_jobs=4, p=pcombo[x][1])
				knn_c.fit(df_train[tcol],df_train['survived'])
				acc = knn_c.score(df_test[tcol],df_test['survived'])

				if acc > bacc:
					bacc = acc
					for z in range(len(pcombo[x])):
						bparam[z] = pcombo[x][z]


	return bparam

def transform():

	f = open("data/train.csv","r")

	#################################
	# features
	#################################

	# survived
	uvec = []
	# class
	lvec = []
	# sex
	svec = []
	# age
	cvec = []
	# age unknown
	kvec = []
	# sibling or spouse
	avec = []
	# parent
	bvec = []

	i = 0
	for line in f:

		if i > 0:

			l = line.split(",")
			uvec.append(int(float(l[1])))
			lvec.append(int(float(l[2])))

			if l[5] == "female":
				svec.append(1)
			else:
				svec.append(0)


			if l[6] != '':
				# age
				cvec.append(int(float(l[6])))
			else: 
				# default 18
				cvec.append(18)
			
			if l[6] == '':
				# age unknown
				kvec.append(1)
			else:
				kvec.append(0)

			if int(float(l[7])) > 0:
				avec.append(1)
			else:
				avec.append(0)

			if int(float(l[8])) > 0:
				bvec.append(1)
			else:
				bvec.append(0)
		else:
			#print line
			#print
			pass

		i += 1

	f.close()

	# write the features to a CSV file and a Numpy array
	train_data = []

	f2 = open("train_2.csv","w")

        f2.write('survived,pclass,sex,age,age_known,sibling/spouse,parent')
        f2.write('\n')
        for x in range(len(uvec)):
                tvec = [uvec[x], lvec[x], svec[x], cvec[x], kvec[x], avec[x], bvec[x]    ]

                tvec_str =  numvec_to_strvec(copy.deepcopy(tvec))

                new_line = ','.join(tvec_str)
                f2.write(new_line)
                f2.write('\n')

                train_data.append(tvec)

        f2.close()


def numvec_to_strvec(vec):

	for x in range(len(vec)):
		vec[x] = str(vec[x])

	return vec	
