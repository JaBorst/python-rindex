import numpy as np
import scipy.sparse as sp
import random

from scipy.signal.ltisys import dimpulse


class IndexVector:

	vec = sp.coo_matrix
	randomIndexDB = {}
	dim = 10
	n = 2


	def __init__(self, dim, n):
		self.dim = dim
		self.n = n

	def createIndexVectorFromContext(self, context=[]):
		#self.vec = tf.SparseTensor(indices=[[0, 0], [1, 2]], values=[1, 2], dense_shape=[3, 4])
		return self.vec

	def createIndexVectorRandom(self, context=[]):
		row = np.array([random.randint(0, self.dim-1) for x in range(self.n)])
		col = np.array([0 for x in range(self.n)])
		values = np.array([random.choice([-1, 1]) for x in range(self.n)])
		self.vec = sp.coo_matrix((values, (row,col)), (self.dim, 1))
		return self.vec

	def set(self, dimension=0, n=0):
		self.dim = dimension
		self.n = n


def main():
	i = IndexVector(100,2)
	i.createIndexVectorRandom()


if __name__ == "__main__":
	main()