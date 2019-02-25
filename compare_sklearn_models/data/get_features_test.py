#!/usr/bin/env python

import numpy as np
import copy

def main():

	print
	print "test"
	print

	f = open("test.csv","r")

	#################################
	# features
	#################################

	# class
	lvec = []

	# sex
	svec = []
	
	# child
	cvec = []

	# sibling or spouse
	avec = []

	# parent
	bvec = []

	i = 0
	for line in f:

		if i > 0:

			l = line.split(",")

			print l
			print

			lvec.append(int(float(l[1])))

			if l[4] == "female":
				svec.append(1)
			else:
				svec.append(0)


			if l[5] != '' and int(float(l[5])) < 18:
				# child
				cvec.append(1)
			elif l[5] != '': 
				# adult
				cvec.append(0)
			else:
				# unknown
				cvec.append(-1)

			if int(float(l[6])) > 0:
				avec.append(1)
			else:
				avec.append(0)

			if int(float(l[7])) > 0:
				bvec.append(1)
			else:
				bvec.append(0)
		else:
			print line
			print

		i += 1


	print
	print "lvec"
	print lvec
	print
	print "svec"
	print svec
	print
	print "cvec"
	print cvec
	print
	print "avec"
	print avec
	print
	print "bvec"
	print bvec
	print

	f.close()

	# write the features to a file
	test_data = []

	f2 = open("test_2.csv","w")

	f2.write('pclass,sex,child,sibling/spouse,parent')
	f2.write('\n')
	for x in range(len(lvec)):
		tvec = [lvec[x], svec[x], cvec[x], avec[x], bvec[x]]

		tvec_str =  numvec_to_strvec(copy.deepcopy(tvec))

		new_line = ','.join(tvec_str)
		f2.write(new_line)
		f2.write('\n')

		test_data.append(tvec)

	f2.close()

	test_data = np.asarray(test_data)

	np.save("test_2_data",test_data)


def numvec_to_strvec(vec):

	for x in range(len(vec)):
		vec[x] = str(vec[x])

	return vec

if __name__ == '__main__':

	main()
