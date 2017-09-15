
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
import numpy as np

import pickle
from operator import mul
from itertools import combinations

import pylab as Plot
from helpers import printProgress


dim = 1000
k = 10


#output
numberOfOnes = 0
numberOfMinusOnes = 0
numberOfZeros = 0
numberOfWords = 0

dictionaryOfPositions = {}
orthogonalAggregates = {}


def generate(inputFile=""):
	global numberOfOnes
	global numberOfMinusOnes
	global numberOfZeros
	global numberOfWords

	global dictionaryOfPositions

	iv = IndexVectorSciPy.IndexVector(dim=dim, n=k)
	ivList = []
	print("Loading Vectors")
	with open(inputFile, 'r') as wordListInput:
		ivList =[iv.create_index_vector_from_context([word]).transpose().toarray()[0] for word in wordListInput]

	print("Laoding Positions")

	for v in ivList:

		unique, counts = np.unique(v, return_counts=True)
		print(unique, counts)
		numberOfOnes += counts[2]
		numberOfMinusOnes += counts[0]
		numberOfZeros += counts[1]

		for pos in np.nonzero(v)[0]:
			dictionaryOfPositions[pos] = dictionaryOfPositions.get(pos, 0) + 1