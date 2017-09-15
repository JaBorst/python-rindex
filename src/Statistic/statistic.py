
import sys
import statistics
import time
import datetime
from numpy.core.operand_flag_tests import inplace_add

sys.path.append('../')
sys.path.append('./')
sys.path.append('Statistic/')
from math import isnan
from  rindex import *
import numpy as np
from scipy.sparse import  find
import pickle
from operator import mul
from itertools import combinations

import pylab as Plot
from helpers import printProgress
##########################
#Statistic Parameters

#input
dim = 1000
k = 100


#output
numberOfOnes = 0
numberOfMinusOnes = 0
numberOfZeros = 0
numberOfWords = 0

dictionaryOfPositions = {}
orthogonalAggregates = {}
#########################





def generateIndexVectors(inputFile=""):
	global numberOfOnes
	global numberOfMinusOnes
	global numberOfZeros
	global numberOfWords

	global dictionaryOfPositions


	print("Loading ",inputFile)
	vectors =[]
	with open (inputFile, "r") as wordListInput:
		words =  wordListInput.readlines()
		print(len(words))

		iv = IndexVectorSciPy.IndexVector(dim=dim, n=k)

		for word in words[:1000]:
			#print(word.replace("\n",""))
			v = iv.create_index_vector_from_context([word.replace("\n","")])
			#print(v)
			vlist =	list(v.transpose().toarray()[0])
			vectors.append(vlist)
			numberOfWords += 1
			numberOfOnes += vlist.count(1)
			numberOfMinusOnes += vlist.count(-1)
			numberOfZeros += vlist.count(0)

			for pos in find(v)[0]:
				dictionaryOfPositions[pos] = dictionaryOfPositions.get(pos,0) + 1
			#break
		with open("files/array.vectors", "wb") as dumpFile:
			pickle.dump(vectors, dumpFile)
		return "files/array.vectors"

def dot(l1 = [], l2 = []):
	return sum(list(map(mul, l1, l2)))



def orthogonality(inputFile):

	with open(inputFile,"rb") as vectorsInputFile:
		vectors = pickle.load(vectorsInputFile)


	global orthogonalAggregates
	i = 0
	numEntries = len(vectors) *len(vectors)/2 -len(vectors)
	printProgress(i, numEntries, prefix='Progress: ' , suffix='Complete: %i' % (i),barLength=50)
	for (v, w) in combinations(vectors, r=2):

		i += 1
		if i % 100 == 0:
			printProgress(i, numEntries, prefix='Progress: ' , suffix='Complete: %i' % (i),barLength=50)

		co=0
		try:
			co = (dot(v, w)/(np.sqrt(dot(v,v))*np.sqrt(dot(w,w)))).round(2)
			if not isnan(co):
				orthogonalAggregates[co] = orthogonalAggregates.get(co, 0) + 1
		except:
			print("Fehler Winkel Berechnung")
			exit()
		#print(co)
		# orthogonalAggregates[co] = orthogonalAggregates.get(co,0)+1




def stats(inputFile =""):
	start = time.time()
	dump = generateIndexVectors(inputFile)
	end = time.time()
	global orthogonalAggregates
	orthogonality(dump)

	with open("statistic.log", "w") as statisticOutput:


		# Probability for a specific position overall
		sumOfChoices = sum(list(dictionaryOfPositions.values()))
		goal = 1./dim
		probabilties = [x/sumOfChoices for x in dictionaryOfPositions.values()]


		statisticOutput.write("Input File: " + inputFile +"\n")

		statisticOutput.write("Configuration:\n")
		statisticOutput.write("\tWordcount:\t" + str(numberOfWords) + "\n")

		statisticOutput.write("\tDimensions:\t " + str(dim) + "\n")
		statisticOutput.write("\tk:\t " + str(k) + "\n")


		statisticOutput.write("Time Creating the Indexvectors:\t" + str(time.strftime("%H:%M:%S", time.gmtime(end-start))) + "\n")



		statisticOutput.write("\nValue Distribution\n")
		statisticOutput.write("\tZeros:\t" + str(numberOfZeros) + "\t P(0)=" + str(numberOfZeros/(numberOfWords*dim)))
		statisticOutput.write("\n\tOnes:\t" + str(numberOfOnes) + "\t P(1)=" + str(numberOfOnes/(numberOfWords*dim)))
		statisticOutput.write("\n\tMinus:\t" + str(numberOfMinusOnes) + "\t P(-1)=" + str(numberOfMinusOnes/(numberOfWords*dim)))


		statisticOutput.write("\n\tP(1| !=0): \t" + str(numberOfOnes/(numberOfOnes+numberOfMinusOnes)))
		statisticOutput.write("\n\tP(-1| !=0): \t" + str(numberOfMinusOnes/(numberOfOnes+numberOfMinusOnes)))


		statisticOutput.write("\n\nPositional Distribution\n")
		statisticOutput.write("Optimal Probability " + str(goal) + "\n")
		statisticOutput.write("\nSampled Data\n")
		statisticOutput.write("\tMean: " + str(statistics.mean(probabilties)) + "\n")
		statisticOutput.write("\tStDev: " + str(statistics.stdev(probabilties)) + "\n")



		Plot.scatter( x = list(dictionaryOfPositions.keys()), y = probabilties)
		Plot.plot((1, 100), (goal,goal), 'k-')
		Plot.savefig("positions.pdf")
		Plot.show()


		statisticOutput.write("\nAngular Distribution (Cosine)\n")
		statisticOutput.write("Optimal: 0\n")
		statisticOutput.write("Sampled angular Data\n")
		# print(orthogonalAggregates['nan'])
		statisticOutput.write("\tMean: " + str(statistics.mean(list(orthogonalAggregates.keys()))) + "\n")
		statisticOutput.write("\tStDev: " + str(statistics.stdev(list(orthogonalAggregates.keys()))) + "\n")
		sumOfChoices = sum(list(orthogonalAggregates.values()))
		goal = 1. / dim
		probabilties = [x / sumOfChoices for x in orthogonalAggregates.values()]

		X = np.array(list(orthogonalAggregates.keys()))
		indexs_to_order_by = X.argsort()
		x_ordered = X[indexs_to_order_by]
		y_ordered = np.array(probabilties)[indexs_to_order_by]

		Plot.scatter(x=x_ordered, y=y_ordered, marker="o")
		Plot.savefig("angles.pdf")
		Plot.show()


def main():

	if sys.argv[1] == "gen":
		print("Generating Wordlist")

		# try:
		generateIndexVectors(sys.argv[2])

	if sys.argv[1] == "ortho":
		print("Generating Wordlist")
		orthogonality(sys.argv[2])


	if sys.argv[1] == "stats":
		print("Generating Wordlist")

		# try:
		stats(sys.argv[2])



if __name__ == "__main__":
	main()


