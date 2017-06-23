import sys
sys.path.append('../')


import itertools
import scipy.sparse as sp
from scipy import spatial
from sklearn.neighbors import KDTree

import numpy as np
import pickle
from rindex import IndexVectorSciPy as IndexVec


import operator
# fix_vec = lambda input:[(number,number+0.0001)[number == 0.0] for
# 						number in input]
import time

class RIModel:
	"""
	RIModel - Random IndexVector Model
	"""
	# das ist eher ungeschickt bei der paralellen verarbeitung.
	# -> keine klassenvariable
	#ContextVectors = {}
	iv = IndexVec.IndexVector
	dim = 0
	k = 0
	index_memory = {} # kann so bleiben
	is_sparse = True

	def __init__(self, dim=0, k=0):
		self.dim = dim
		self.k = k
		self.iv = IndexVec.IndexVector(dim, k)
		self.ContextVectors = {}
		self.word_count = {}



	def write_model_to_file(self, filename = "dump.model"):
		"""
		:param filename:
		:return:
		"""
		print("write model to ",filename)
		with open(filename, 'wb') as output:
			pickle.dump(self.ContextVectors, output)


	def load_model_from_file(self, filename):
		"""
		loads saved context-vectors. also sets is_sparse and dim
		:param filename:
		:return:
		"""
		with open(filename, 'rb') as inputFile:
			self.ContextVectors = pickle.load(inputFile)
		# get any element and determine if it's sparse or not
		some_element = self.ContextVectors[next(iter(self.ContextVectors))]

		if sp.issparse(some_element):
			self.is_sparse = True
			self.dim = some_element.shape[0]
		else:
			self.is_sparse = False
			self.dim = len(some_element)




	def add_document(self, context=[]):
		"""     Takes the Context array as the context of its entries and each element of the array as word
		"""
		for word in context:
			if word in self.ContextVectors.keys():
				pass
			else:
								self.ContextVectors[word] = sp.coo_matrix((self.dim, 1))
			self.ContextVectors[word] += self.iv.create_index_vector_from_context(word)

	def add_context(self, context = [], index = 0):
		"""Add a self defined Context for a specifix word with index , possibly with a weight mask
		   default: index = 0 ( the first word in the array ) , if no
		mask given all contexts are weighted 1
		:param context[]:
		:param index:
		:param mask:
		"""


		if context[index] in self.ContextVectors.keys():
			pass
		else:
			self.ContextVectors[context[index]] = sp.coo_matrix((self.dim, 1))

		rest = context[:index] + context[index+1:]#-> should not be done here
		self.ContextVectors[context[index]] += self.iv.create_index_vector_from_ordered_context(rest)

		# for word, weight in zip(context, mask):
		# 	if word == context[index]:
		# 		continue
		# 	if context[index] not in self.ContextVectors.keys():
		# 		self.ContextVectors[context[index]] = sp.coo_matrix((self.dim, 1))
		# 	if word not in self.index_memory.keys():
		# 		self.index_memory[word] = self.iv.create_index_vector_from_context([word])
		# 	self.ContextVectors[context[index]] += self.index_memory[word] * weight


	def add_unit(self, unit="", context=[], weights=[]):
		"""     Explicitly Specify the unit of interest and and the units of context.
				That way you can specify a document as unit of interest and the contained words as "contexts" for example
		"""
		#check if enough weights present:
		if len(weights)!=0 and len(context) != len(weights):
			raise ValueError("Not Matching Array Lengths in addUnit()")
		elif len(weights) == 0:
			weights = [1] * len(context)


		#check if the unit is already available if not create vector
		if unit in self.ContextVectors.keys():
			pass
		else:
			self.ContextVectors[unit] = sp.coo_matrix((self.dim, 1))

		#add every context unit to the unit of interest (and save to memory for now)
		for entry in context:
			if entry not in self.index_memory.keys():
			 	self.index_memory[entry] = self.iv.create_index_vector_from_context([entry])

			self.ContextVectors[unit] += self.index_memory[entry]*(weights[context.index(entry)])

			#self.ContextVectors[unit] += self.iv.create_index_vector_from_context([entry]) * weights[context.index(entry)]




	def get_similarity_cos(self, word1, word2):
		"""    Calculate the Cosine-Similarity between the two elements word1 and word2
		 word1, word2 must occur in the Model
		 :param word1:
		 :param word2:
		 :return: cosinus-distance
			 """
		 # 1- ist bei mir so, kann man aber auch einfach ändern.
		if self.is_sparse:
			return 1-spatial.distance.cosine(self.ContextVectors[word1].toarray(),
											 self.ContextVectors[word2].toarray())
		else:
			return 1 - spatial.distance.cosine(self.ContextVectors[word1],
											   self.ContextVectors[word2])


	def get_similarity_jaccard(self, word1 = "", word2 = ""):
		"""Calculate the Dice coeffiecient
		http://nlp.ffzg.hr/data/publications/nljubesi/ljubesic08-comparing.pdf
		"""
		featureMins = 0
		featureMax = 0
		cv1 = np.array(self.ContextVectors[word1].toarray().transpose()[0])
		cv2 =  np.array(self.ContextVectors[word2].toarray().transpose()[0])

		pos1 = np.append(np.abs(cv1.clip(max=0)), cv1.clip(min=0))
		pos2 = np.append(np.abs(cv2.clip(max=0)), cv2.clip(min=0))

		featureMins = np.sum(np.minimum(pos1, pos2))
		featureMax = np.sum(np.maximum(pos1, pos2))

		return featureMins/featureMax

	def entropy(self, v=np.array([])):
		_Hv = (v * np.log2(v))
		_Hv[_Hv != _Hv] = 0
		_Hv[np.abs(_Hv) == np.inf] = 0
		return _Hv

	def kld(self, v=np.array([]), w=np.array([])):
		np.seterr(divide='ignore', invalid='ignore')
		_kld = v * np.log2(np.divide(v, w))
		_kld[_kld != _kld] = 0
		np.seterr(divide='warn', invalid='warn')
		return np.sum(_kld)

	def get_similarity_jsd(self, word1, word2):
		""" Jensen Shannon Entropy as Similarity (hence return 1- ...)

		:param word1: 
		:param word2: 
		:return: 
		"""
		# Get the Context Vectors
		cv1 = self.ContextVectors[word1].toarray().transpose()[0]
		cv2 = self.ContextVectors[word2].toarray().transpose()[0]

		# Extract the Probabilty Dimensions
		# First Positive then negative
		_P = np.zeros(2 * self.dim)
		sum = np.sum(cv1.clip(min=0)) - np.sum(cv1.clip(max=0))
		for i in range(0, self.dim):
			if cv1[i] > 0:
				_P[i] = cv1[i] / sum
			elif cv1[i] < 0:
				_P[i + self.dim] = -cv1[i] / sum

		_Q = np.zeros(2 * self.dim)
		sum = np.sum(cv2.clip(min=0)) - np.sum(cv2.clip(max=0))
		for i in range(0, self.dim):
			if cv2[i] > 0:
				_Q[i] = cv2[i] / sum
			elif cv2[i] < 0:
				_Q[i + self.dim] = -cv2[i] / sum

		_M = 0.5 * (_P + _Q)

		return 1 - 0.5 * np.sum(self.kld(_P, _M) + self.kld(_Q, _M))

	def get_similarity(self, word1 = "", word2 = "", method = "cos"):
		"""

		:param word1:
		:param word2:
		:param method: defaults to cosine
		:return:
		"""
		if method == "cos":
			return self.get_similarity_cos(word1,word2)
		if method == "jaccard":
			return self.get_similarity_jaccard(word1,word2)
		if method == "jsd":
			return self.get_similarity_jsd(word1,word2)

	def is_similar_to(self, word ="", thres = 0.9, count = 10):
		""" Returns words with the least distance to the given word.
		The combination of threshold and count can be used e.g. for
		testing to get a small amount of words (and stop after that).
		:param word:
		:param thres:
		:param count
		:return list of similar words
		"""
		if len(word) == 0:
			return
		try:
			# check if ContextVector exists
		    self.ContextVectors[word]
		except:
			print("Word not in context")
			return
		i = 0
		start = time.time()
		max_sim=0.0
		max_word=""

		sum_time = 0.0
		for i in range(1):
			start = time.time()
			for key in self.ContextVectors.keys():
				if key != word:
					sim = self.get_similarity(word, key, method=method)

					if max_sim<sim:
						max_sim = sim
						max_word = key
						print(key, sim)

			sum_time += time.time()-start
		#print(sum_time/1,method)
		#print("max",max_word, max_sim)
		#print("searching original structure for {0} took me {1} sec.".format(count, time.time() - start))

					# hier geht was schief, muss aber auch nich unbedingt
		# if i < count:
		#     print("\n\n {0} most similar words".format(count) )
		#     best_n = dict(sorted(sims.items(), key=operator.itemgetter(1), reverse=True)[:count])
		#     for word in best_n.keys():
		#         print(word,"\t\t", best_n[word])



	def to_matrix(self, to_sparse=False):
		"""
		:param to_sparse:
		 :param: is_sparse:
		:return: 
		"""
		keys = self.ContextVectors.keys()
		i = range(len(keys))
		if to_sparse and self.is_sparse:
			target_martix = sp.lil_matrix((len(keys), self.dim))
			for key, i in zip(keys, i):
				target_martix[i] = self.ContextVectors[key].transpose()
		elif not to_sparse:
			target_martix = np.zeros((len(keys), self.dim))
			if self.is_sparse:
				for key, i in zip(keys, i):
					target_martix[i, :] = self.ContextVectors[key].transpose().toarray()
			else:
				for key, i in zip(keys, i):
					target_martix[i, :] = self.ContextVectors[key]
		else:
			print('invalid option')
		print("converted dict to matrix")
		return list(keys), target_martix


	def to_tree(self, method='minkowski', leaf_size = 50):
		"""
		should be done !only! with reduced data
		to enhence search. for ways to search check readTest.py
		:param method: 
		:return: 
		"""
		if self.dim > 50:
			print(">50 dim is not recommended. please reduce first.")
			return
		keys, leafs = self.to_matrix(to_sparse=False)
		return keys, KDTree(leafs, leaf_size=leaf_size, metric=method)


	def reduce_dimensions(self,method= "random_projection", target_size=100):
		"""
		1) call to_matrix 2) reduce dim 3) fill to new dict
		:param method:
		:param target_size:
		:return:
		"""

		if method == "random_projection" or method == "truncated_svd":
			keys, target_matrix = self.to_matrix(to_sparse=True)

		elif method == "mds" or method == "tsne":
			keys, target_matrix = self.to_matrix(to_sparse=False)
			target_size = 2

		print("Reduce ", target_matrix.shape[1], " to ", target_size)
		if method == "random_projection":
			"""
				SPARSE_RANDOM_PROJECTION
			"""
			print("using SparseRandomProjection...")
			from sklearn.random_projection import SparseRandomProjection
			sparse = SparseRandomProjection(n_components=target_size)
			red_data = sparse.fit_transform(target_matrix)
			self.is_sparse = True

		elif method == "truncated_svd":
			"""
				TRUNCATED_SVD
			"""
			from sklearn.decomposition import TruncatedSVD
			print("using TruncatedSVD...")
			svd = TruncatedSVD(n_components=target_size, n_iter=10, random_state=42)
			red_data = svd.fit_transform(target_matrix)
			print("sd-sum is:\t", svd.explained_variance_ratio_.sum())
			self.is_sparse = False

		elif method == "mds":
			"""
				MDS: ist leider recht ineffizient implementiert...
			"""
			from sklearn import manifold
			print("use mds...good luck with that!")
			seed = np.random.RandomState(seed=3)
			mds = manifold.MDS(n_components=2, max_iter=10, eps=1e-6, random_state=seed,
							dissimilarity="euclidean", n_jobs=2, verbose=1)
			red_data = mds.fit_transform(target_matrix)#.embedding_
			self.is_sparse = False

		elif method == "tsne":
			from sklearn.manifold import TSNE
			print("use tsne")
			model = TSNE(n_components=2, random_state=0, metric='euclidean')
			np.set_printoptions(suppress=True)
			red_data = model.fit_transform(target_matrix)
			self.is_sparse = False
		self.ContextVectors = {}  # reset dicct

		for i, key in zip(range(len(keys)),keys):
				self.ContextVectors[key] = red_data[i]
		self.dim = target_size

	def vector_add(self, words =[], isword="" ):
		"""
		check if sum of words is equal isword
		:param words:
		:param isword:
		:return: distance from sum to actual word
		"""
		if self.is_sparse:
			iv_sum = sp.csr_matrix((self.dim, 1))
			for word in words:
				iv_sum += self.ContextVectors[word]
			return 1 - spatial.distance.cosine(iv_sum.toarray(),
										self.ContextVectors[isword].toarray())
		elif not self.is_sparse:
			iv_sum = np.zeros((self.dim, 1))
			for word in words:
				iv_sum += self.ContextVectors[word][0]
			return 1-spatial.distance.cosine(iv_sum,
										self.ContextVectors[isword])


	def most_similar(self, count=10, file=None):
		""" Compare all words in model. (Takes long Time)
		Isn't that one reason why we need dim-reduction?
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
			i += 1
			print("\r%f %%" % (100*i/size), end="")
			simDic.append([pair[0], pair[1], self.get_similarity_cos(pair[0], pair[1])])
		print("Sorting...")
		simDic.sort(key=lambda x: x[2], reverse=True)
		for x in range(count):
			print(x, ":\t", simDic[x])
		if file:
			with open(file, 'w') as out:
				for triple in simDic:
					out.write(triple[0] + "\t" + triple[1] + "\t" + str(triple[2]) + "\n")


def main():
	"""Main function if the Module gets executed"""
	dim = 1000
	k = 3
	r = RIModel(dim, k)
	r.add_context(["hello", "world", "damn"])

	r.add_context(["hello", "world", "damn"], index=0, mask=[0, 0.5, 0.5])
	r.add_context(["hello", "world", "example"], index=0, mask=[0, 0.5, 0.5])
	r.add_context(["hello", "world", "damn"], index=0, mask=[0, 0.5, 0.5])
	r.add_context(["hello", "world", "example"], index=0, mask=[0, 0.5, 0.5])
	r.add_context(["hello", "world", "example"], index=0, mask=[0, 0.5, 0.5])
	r.add_context(["hello", "damn", "nice"], index=0, mask=[0, 0.5, 0.5])

	r.add_context(["the", "damn", "example"], index=0, mask=[0, 0.5, 0.5])
	r.add_context(["the", "world", "example"], index=0, mask=[0, 0.5, 0.5])
	r.add_context(["the", "world", "example"], index=0, mask=[0, 0.5, 0.5])
	r.add_context(["the", "world", "example"], index=0, mask=[0, 0.5, 0.5])
	r.add_context(["the", "world", "nice"], index=0, mask=[0, 0.5, 0.5])
	r.add_context(["the", "world", "nice"], index=0, mask=[0, 0.5, 0.5])

	r.add_context(["the", "world", "damn"], index=0, mask=[0, 0.5, 0.5])
	r.add_context(["parks", "are", "shitty"], index=0, mask=[0, 0.5, 0.5])

	# r.writeModelToFile()
	# rmi = RIModel(dim, 3)
	# filename = "/home/tobias/Dokumente/saved_context_vectors/d100k3.pkl"
	# rmi.loadModelFromFile(filename)


	print("JSD: ", r.get_similarity_jsd("hello", "parks"))
	print("Cos: ", r.get_similarity_cos("hello", "parks"))
	print("JACC: ",r.get_similarity_jaccard("hello", "parks"))


	print("JSD: ", r.get_similarity_jsd("hello", "hello"))
	print("Cos: ", r.get_similarity_cos("hello", "hello"))
	print("JACC: ",r.get_similarity_jaccard("hello", "hello"))
	#r.is_similar_to(word ="hello", thres = 0.1, count = 10)
	#print(r.vector_add(words=["the","parks"],isword="hello"))

if __name__ == "__main__":
	main()
