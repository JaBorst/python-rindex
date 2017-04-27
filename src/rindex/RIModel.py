import itertools

import scipy.sparse as sp
import numpy as np
from rindex import IndexVectorSciPy as IndexVec

class RIModel:

	ContextVectors = {}
	iv = IndexVec.IndexVector
	dim = 0
	k = 0

	index_memory = {}

	def __init__(self, dim=0, k=0):
		self.dim = dim
		self.k = k
		self.iv = IndexVec.IndexVector(dim, k)


	def addDocument(self,context=[]):
		""" 	Takes the Context array as the the context of its entries and each element of the array as word
		"""
		for word in context:
			if word in self.ContextVectors.keys():
				self.ContextVectors[word] += self.iv.createIndexVectorRandom()
			else:
				self.ContextVectors[word] = sp.coo_matrix((self.dim, 1))
				self.ContextVectors[word] += self.iv.createIndexVectorRandom()


	def addContext(self, context = [], index = 0, mask = None):
		"""Add a self defined Context for a specifix word with index , possibly with a weight mask
		   default: index = 0 ( the first word in the array ) , if no mask given all contexts are weighted 1"""

		if not mask:
			mask = [1] * len(context)

		for word, weight in zip(context, mask):
			if word == context[index]:
				continue

			if word not in self.index_memory.keys():
				self.index_memory[word] = self.iv.createIndexVectorRandom()

			if context[index] in self.ContextVectors.keys():
				self.ContextVectors[context[index]] += weight * self.index_memory[word]
			else:
				self.ContextVectors[context[index]] = sp.coo_matrix((self.dim, 1))
				self.ContextVectors[context[index]] += weight * self.index_memory[word]



	def getSimilarityCos(self, word1, word2):
		"""	Calculate the Cosine-Similarity between the two elements word1 and word2
			word1, word2 must occur in the Model
		:param word1:
		:param word2:
		:return:
		"""

		scalarproduct = (self.ContextVectors[word1].transpose() * self.ContextVectors[word2]).toarray()[0][0]
		norm1 = (self.ContextVectors[word1].transpose() * self.ContextVectors[word1]).toarray()[0][0]
		norm2 = (self.ContextVectors[word2].transpose() * self.ContextVectors[word2]).toarray()[0][0]
		return scalarproduct / (np.sqrt(norm1) * np.sqrt(norm2))


	def mostSimilar(self,count= 10, file = None ):
		""" Compare all words in model. (Takes long Time)
		:param count:
		:param file:
		:return:
		"""

		simDic = []
		keys = self.ContextVectors.keys()
		tuples = list(itertools.combinations(keys, 2))
		print("Comparing everything...")

		i = 0
		size = len(tuples)
		for pair in tuples:
			i+=1

			print("\r%f %%" % (100*i/size), end="")
			simDic.append([pair[0], pair[1], self.getSimilarityCos(pair[0], pair[1])])

		print("Sorting...")
		simDic.sort(key=lambda x: x[2], reverse=True)
		for x in range(count):
			print(x, ":\t", simDic[x])

		if file:
			with open(file, 'w') as out:
				for triple in simDic:
					out.write(triple[0] + "\t" + triple[1] + "\t" + str(triple[2]) + "\n")


def main():
	"""Main function if the MOdule gets executed"""
	r = RIModel(20, 3)

	r.addContext(["hello", "world", "damn"], index=0, mask=[0, 0.5, 0.5])
	r.addContext(["hello", "world", "example"], index=0, mask=[0, 0.5, 0.5])
	r.addContext(["hello", "world", "damn"], index=0, mask=[0, 0.5, 0.5])
	r.addContext(["hello", "world", "example"], index=0, mask=[0, 0.5, 0.5])
	r.addContext(["hello", "world", "example"], index=0, mask=[0, 0.5, 0.5])
	r.addContext(["hello", "damn", "shitty"], index=0, mask=[0, 0.5, 0.5])

	r.addContext(["the", "damn", "example"], index=0, mask=[0, 0.5, 0.5])
	r.addContext(["the", "world", "example"], index=0, mask=[0, 0.5, 0.5])
	r.addContext(["the", "world", "example"], index=0, mask=[0, 0.5, 0.5])
	r.addContext(["the", "world", "example"], index=0, mask=[0, 0.5, 0.5])
	r.addContext(["the", "world", "shitty"], index=0, mask=[0, 0.5, 0.5])
	r.addContext(["the", "world", "shitty"], index=0, mask=[0, 0.5, 0.5])

	print(r.ContextVectors)

	for key in r.index_memory.keys():
		print(key, ":\t", r.index_memory[key].toarray())

	print(r.getSimilarityCos("hello", "the"))

if __name__ == "__main__":
	main()