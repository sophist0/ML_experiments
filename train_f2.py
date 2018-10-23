import numpy as np
import copy

def get_features():

	fmat = read_features()
	train_cf = custom_f(fmat)

	return train_cf

def read_features():

	f = open("train.csv","r")

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


			#if l[6] != '' and int(float(l[6])) < 18:
			#	# child
			#	cvec.append(1)
			#elif l[6] != '': 
			#	# adult
			#	cvec.append(-1)
			#else:
			#	# unknown
			#	cvec.append(0)

			if l[6] != '': 
				# adult
				cvec.append(18 - int(float(l[6])))
			else:
				# unknown
				cvec.append(0)


			if int(float(l[7])) > 0:
				avec.append(1)
			else:
				avec.append(0)

			if int(float(l[8])) > 0:
				bvec.append(1)
			else:
				bvec.append(0)
		else:
			pass
			#print line
			#print

		i += 1

	f.close()

	#uvec = np.asarray(uvec) 
	lvec = np.asarray(lvec) 
	svec = np.asarray(svec) 
	cvec = np.asarray(cvec) 
	avec = np.asarray(avec) 
	bvec = np.asarray(bvec)

	#uvec2 = norm_vec(uvec)
	lvec2 = norm_vec(lvec)
	svec2 = norm_vec(svec)
	cvec2 = norm_vec(cvec)
	avec2 = norm_vec(avec)
	bvec2 = norm_vec(bvec)

	print
	print lvec2.mean()
	print lvec2.std()
	print

	return [uvec, lvec2, svec2, cvec2, avec2, bvec2]

def norm_vec(vec):

	v_ave = vec.mean()
	v_std = vec.std()

	vec2 = (vec - v_ave) / float(v_std)

	return vec2

def custom_f(fmat):

	uvec = fmat[0]
	lvec = fmat[1]
	svec = fmat[2]
	cvec = fmat[3]
	avec = fmat[4]
	bvec = fmat[5]

	# write the features to a CSV file and a Numpy array
	train_data = []

	f2 = open("train_2.csv","w")

	f2.write('survived,pclass,sex,child,sibling/spouse,parent')
	f2.write('\n')
	for x in range(len(uvec)):
		tvec = [uvec[x], lvec[x], svec[x], cvec[x], avec[x], bvec[x]]

		tvec_str =  numvec_to_strvec(copy.deepcopy(tvec))

		new_line = ','.join(tvec_str)
		f2.write(new_line)
		f2.write('\n')

		train_data.append(tvec) 

	f2.close()

	return train_data

def numvec_to_strvec(vec):

	for x in range(len(vec)):
		vec[x] = str(vec[x])

	return vec


