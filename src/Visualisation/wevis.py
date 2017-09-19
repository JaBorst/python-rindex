import sys

from sklearn.manifold.tests.test_isomap import path_methods

sys.path.append("../")
import traceback
import matplotlib.pyplot as plt
import numpy as np
import itertools
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import normalize
from sklearn.neighbors import DistanceMetric

testpoints = ["König","Königin", "Gräfin", "Graf", "Kaiser", "Kaiserin", "Gemahlin", "Gemahl", "Äbtissin","Abt", "Prinz", "Prinzessin", "Mann", "Frau", "Freund", "Freundin"]
testpairs = [
	  ["König","Königin"]
	, ["Graf", "Gräfin"]
	, ["Kaiser", "Kaiserin"]
	, ["Gemahl", "Gemahlin"]
	, ["Abt", "Äbtissin"]
	, ["Prinz", "Prinzessin"]
	, ["Mann", "Frau"]
	, ["Freund","Freundin"]
	, ["Partner", "Partnerin"]
	, ["Stellvertreter", "Stellvertreterin"]
	][:8]
addWords = {'positive': ["Frau"], 'negative':["Mann"]}


class Wevis:

	def __init__(self):
		self.word_vectors = []
		self.dictionary = {}
		self.inverse_dictionary = {}
		self.model = 'name'
		self.savepath = "./"
		self.reduced_data= None
		self.reduced_data_method = ""
		self.reduced_data_method_metric = ""

	def set_title(self, t=""):
		self.model = t
	def set_savepath(self, s=""):
		self.savepath=s

	def load_model(self, filepath="", norm = True):
		if filepath == "" :
			print("Please hand over filepath...")
			exit()
		else:
			import pickle
			with open (filepath,"rb") as inputFile:
				wd = pickle.load(inputFile)
				if norm:
					self.word_vectors = normalize(   list(wd.values()), axis=0)
				else:
					self.word_vectors = list(wd.values())

				self.dictionary = dict(zip(wd.keys(),range(len(wd.keys()))))


				if hasattr(self.dictionary, 'iteritems'):
					# Python 2 compat
					items_iterator = self.dictionary.iteritems()
				else:
					items_iterator = self.dictionary.items()

				self.inverse_dictionary = {v: k for k, v in items_iterator}

	def info(self):
		print("Wevis - Word Embedding Visualisation")
		print("Containing %i Words" % (len(self.word_vectors)))
		print(self.word_vectors[0].shape)
		#print(self.dictionary)

	def dim_reduce(self, method = "tsne", target_dim = 2, points = None, metric = "minkoswki"):

		try:
			if len(self.reduced_data) != 0 :
				if self.reduced_data_method == method and method != "isomap":
					return self.reduced_data
				elif method == "isomap" and self.reduced_data_method== method:
					if self.reduced_data_method_metric == metric:
						return self.reduced_data
		except:
			pass

		if method == "tsne":

			from sklearn.manifold import TSNE
			tsne = TSNE(n_components=target_dim, random_state=42)
			np.set_printoptions(suppress=True)

			self.reduced_data_method_metric = ""
			self.reduced_data_method = "tsne"
			if points == None:
				self.reduced_data = tsne.fit_transform(self.word_vectors[:1000])
			else:
				self.reduced_data = tsne.fit_transform(points)


		elif method == "truncated_svd":
			from sklearn.decomposition import TruncatedSVD
			print("using TruncatedSVD...")
			svd = TruncatedSVD(n_components=target_dim, n_iter=10, random_state=42)
			self.reduced_data_method_metric = ""
			self.reduced_data_method = "truncated_svd"
			if points == None:
				self.reduced_data = svd.fit_transform(self.word_vectors[:1000])
			else:
				self.reduced_data = svd.fit_transform(points)
			print("sd-sum is:\t", svd.explained_variance_ratio_.sum())

		elif method == "spectral":
			from sklearn.manifold import SpectralEmbedding
			se = SpectralEmbedding(n_components=target_dim, random_state=42)
			self.reduced_data_method_metric = ""
			self.reduced_data_method = "spectral"
			if points == None:
				self.reduced_data = se.fit_transform(self.word_vectors[:1000])
			else:
				self.reduced_data = se.fit_transform(points)

		elif method == "isomap":
			from sklearn.manifold.isomap_mod import Isomap
			i = Isomap(n_components=target_dim, max_iter= 1000, path_method='D', neighbors_algorithm='auto')
			self.reduced_data_method_metric = metric
			self.reduced_data_method = "isomap"
			if points == None:
				self.reduced_data = i.fit_transform(self.word_vectors[:1000], metric=metric)
			else:
				self.reduced_data = i.fit_transform(points, metric=metric)

		elif method == "lle":
			from sklearn.manifold import LocallyLinearEmbedding
			lle = LocallyLinearEmbedding(n_components=target_dim, max_iter=1000, neighbors_algorithm='auto')
			self.reduced_data_method_metric = ""
			self.reduced_data_method = "lle"
			if points == None:
				self.reduced_data = lle.fit_transform(self.word_vectors[:1000])
			else:
				self.reduced_data = lle.fit_transform(points)

		elif method == "kpca":
			from sklearn.decomposition import PCA, KernelPCA
			kpca = KernelPCA(kernel="rbf", fit_inverse_transform=True, gamma=10)
			self.reduced_data_method_metric = ""
			self.reduced_data_method = "kpca"
			if points == None:
				self.reduced_data = kpca.fit_transform(self.word_vectors[:1000])
			else:
				self.reduced_data = kpca.fit_transform(points)

		return self.reduced_data


	def wordnet(self, word = "", method = "tsne", num = 10 ):
		print("new2")
		first_level = self.glove_similarity_query(word,num)
		word_collection_first_level = [w[0] for w in first_level]
		second_level = {}
		word_collection_second_level = []
		connection_dict_first_level = []
		connection_dict_second_level = []

		for (w,s) in first_level:
			second_level[w]= self.glove_similarity_query(w,num)
			connection_dict_first_level.append((w,word))
			for (w2,s2) in second_level[w]:
				connection_dict_second_level.append((w,w2))
				word_collection_second_level.append(w2)


		nodes = list(set(word_collection_first_level + word_collection_second_level))
		print(nodes)
		points = []
		for w in nodes:
			points.append(self.word_vectors[self.dictionary[w]])
		#print(points)

		Y = self.dim_reduce(method=method, target_dim=2,points= points)
		plt.plot(0,0)
		for i, w in enumerate(nodes):

			if w == word:
				print(Y[i, 0], Y[i, 1])
				plt.plot(Y[i, 0], Y[i, 1],'ro')
			elif w in word_collection_first_level:
				plt.plot(Y[i, 0], Y[i, 1],'o', color="#934782")
			elif w in word_collection_second_level:
				plt.plot(Y[i, 0], Y[i, 1],'o', color="#134712")


		for label, x, y in zip(nodes, Y[:, 0], Y[:, 1]):
			plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points', color ="#ee8d18")


		for (key,value )in connection_dict_first_level:
			k_point = Y[nodes.index(key),:]
			v_point = Y[nodes.index(value),:]
			plt.plot([k_point[0], v_point[0]], [k_point[1],v_point[1]], color="0.75")

		#
		# for (key,value )in connection_dict_second_level:
		# 	k_point = Y[nodes.index(key),:]
		# 	v_point = Y[nodes.index(value),:]
		# 	plt.plot([k_point[0], v_point[0]], [k_point[1],v_point[1]], color="0.25")
		plt.show()

	def wordlist(self, words = [], method = "tsne", metric = "manhattan"):

		if len(words) == 0:
			words = testpoints

		nodes = list(set(words))
		print(nodes)
		points = []
		for w in nodes:
			points.append(self.word_vectors[self.dictionary[w]])
		#print(points)

		Y = self.dim_reduce(method=method, target_dim=2, points=points, metric=metric)
		plt.plot(0,0)
		for i, w in enumerate(nodes):
				plt.plot(Y[i, 0], Y[i, 1],'o', color="#134712")


		for label, x, y in zip(nodes, Y[:, 0], Y[:, 1]):
			plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points', color ="#ee8d18")

		plt.show()

	def wordlistwhole(self, words = [], method = "tsne", metric = "manhattan"):

		if len(words) == 0:
			words = testpoints

		nodes = list(set(words))
		print(nodes)
		points = []
		Y = self.dim_reduce(method=method, metric=metric, target_dim=2, points=self.word_vectors)
		for w in nodes:
			points.append(Y[self.dictionary[w]])
			# print(points)
		#print(points)


		plt.plot(0,0)
		for i, w in enumerate(nodes):
				plt.plot(Y[i, 0], Y[i, 1],'o', color="#134712")


		for label, x, y in zip(nodes, Y[:, 0], Y[:, 1]):
			plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points', color ="#ee8d18")

		plt.show()

	def wordpairs(self, words = testpairs, testaddwords=addWords, method = "tsne", metric = "euclidean", save= False, show = True):
		"""
		Take a Sample of Wordpairs and visualize them on a 2D-grid. for test you can spacify a vector dictionary that contains vectors to add and subtract from every first member of a Pair and compairs the result to the second member
		:param words: List of wordpairs like [ ["König", Königin"], ["Graf", "Gräfin"]...]
		:param testaddwords: Dictionary of words to add and subtract like  testaddwords['positive'] = ["Frau"], testaddwords['negative'] = ["Mann"], so it takes the first word of every pair in words and adds the vector of "Frau" and subtracts thevector of "Mann" and compares the result to the second word in every wordpair
		:param method: the method for dimension Reduction
		:param metric: a Metric can only be changed for isomap
		:param save: if set to True the plot will be saved to disk( you can use savepath(..) and title(..) to set the corresponding values)
		:param show: if set to True the plot will be shown
		:return:
		"""



		nodes = list(set([x for sublist in  words for x in sublist]))


		#print(nodes)
		points = []
		for w in nodes:
			points.append(self.word_vectors[self.dictionary[w]])
		Y = self.dim_reduce(method=method, target_dim=2,points= points, metric =  metric)
		plt.clf()
		plt.plot(0,0)
		plt.figure(figsize=( 11.69, 8.27  ))
		if method == 'isomap':

			plt.suptitle("%s - Wordpairs\n%s - %s" % (self.model, method, metric))
		else:
			plt.suptitle("%s - Wordpairs\n%s" % (self.model, method))


		for i, w in enumerate(nodes):
				plt.plot(Y[i, 0], Y[i, 1],'o', color="#134712")


		for label, x, y in zip(nodes, Y[:, 0], Y[:, 1]):
			plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points', color ="#ee8d18")


		for l in words:
			print(l)
			for (key, value) in itertools.combinations(l,2):
				k_point = Y[nodes.index(key), :]
				v_point = Y[nodes.index(value), :]
				plt.plot([k_point[0], v_point[0]], [k_point[1], v_point[1]], color="0.75")




				ranking = [x for x, y in self.glove_similarity_query({'positive': [key]+testaddwords['positive'], 'negative': testaddwords['negative']},number=1000)]

				rank = 0
				try:
					rank = ranking.index(value) +1
				except:
					rank = -1

				if rank != None:
					# print(v_point)
					plt.annotate(rank, xy=(v_point[0], v_point[1]), xytext=(0,-12), textcoords='offset points', color ="#000000")

		if save:
			if method == "isomap":
				plt.savefig("%s/%s-%s(%s).pdf" % ( self.savepath, self.model, method, metric), dpi=(1000), paper='a4')

			else:

				plt.savefig("%s/%s-%s.pdf" % ( self.savepath, self.model, method), dpi=(1000), paper='a4')

		if show:
			plt.show()

	def visualize_vecs(self, vec_names=[]):
		"""
		get a graphic representation of cvs (for testing)
		:param rmi:
		:param vec_names:
		:return:
		"""
		ivs = []
		if len(vec_names) == 0:
			print("no vecs specified. good luck.")
			for name1, name2 in testpairs[:5]:
				vec_names.append(name1+"-Mann+Frau")
				ivs.append(self.vector_add(positive=[name1, "Frau"], negative=["Mann"] ))

				vec_names.append(name2)
				ivs.append(self.vector_add(positive=[name1]))


			print(vec_names)

		else:
			for name in vec_names:
				pass

		from matplotlib import collections as matcoll

		x = [i for i in range(ivs[0].shape[0])]
		f, axarr = plt.subplots(len(vec_names), sharex=True)
		for i in range(len(vec_names)):

			lines = []
			for j in range(len(ivs[i])):
				pair = [(j, 0), (j, ivs[i][j])]
				print(pair)
				lines.append(pair)

			linecoll = matcoll.LineCollection(lines)
			axarr[i].add_collection(linecoll)
			# axarr[i].ylim([-500, 500])
			axarr[i].set_title(vec_names[i])
			axarr[i].set_ylabel('xvalue')

			# plt.ylim([-1, 1])
		axarr[-1].set_xlabel('vec position')
		for sub_plot_axis in plt.gcf().get_axes():
			sub_plot_axis.set_xlim(0,ivs[0].shape[0])
			sub_plot_axis.set_ylim(-100, 100)
		plt.show()

	def wordcloud(self, word, num):
		from wordcloud import WordCloud
		wc = WordCloud()
		print(dict(self.glove_similarity_query(word,num)))
		wc.generate_from_frequencies(dict(self.glove_similarity_query(word,num)))

		plt.figure()
		plt.imshow(wc, interpolation="bilinear")
		plt.axis("off")
		plt.show()

	def vector_add(self, positive=[], negative=[]):
		v = np.array(0)
		for w in positive:
			try:
				v = v + self.word_vectors[self.dictionary[w]]
			except:
				pass
		for w in negative:
			try:
				v = v - self.word_vectors[self.dictionary[w]]
			except:
				pass

		#print(v.shape)
		return v

	def get_similarity(self, word1 = "", word2 = "", method = "cos"):
		"""

		:param word1:
		:param word2:
		:param method: defaults to cosinetolist
		:return:
		"""

		cv1 = self.word_vectors[self.dictionary[word1]]
		cv2 = self.word_vectors[self.dictionary[word2]]

		if method == "cos":
			return cosine(cv1,cv2)

		if method == "jaccard":

			pos1 = np.append(np.abs(cv1.clip(max=0)), cv1.clip(min=0))
			pos2 = np.append(np.abs(cv2.clip(max=0)), cv2.clip(min=0))
			featureMins = np.sum(np.minimum(pos1, pos2))
			featureMax = np.sum(np.maximum(pos1, pos2))

			return featureMins / featureMax

	def glove_similarity_query(self, word, number=10):
		if type(word) is str:
			word_vec = self.word_vectors[self.dictionary[word]]
		elif type(word) is dict:
			word_vec = self.vector_add(positive=word["positive"], negative=word["negative"])

		else:
			print("Not a valid type")
			return []

		dst = (np.dot(self.word_vectors, word_vec)
		       / np.linalg.norm(self.word_vectors, axis=1)
		       / np.linalg.norm(word_vec))
		word_ids = np.argsort(-dst)
		# print(dst.shape, word_ids)
		return [(self.inverse_dictionary[x], dst[x]) for x in word_ids[:number]
		        if x in self.inverse_dictionary]

	def distance_similarity_query(self, word, number=10):
		if type(word) is str:
			word_vec = self.word_vectors[self.dictionary[word]]
		elif type(word) is dict:
			word_vec = self.vector_add(positive=word["positive"], negative=word["negative"])

		else:
			print("Not a valid type")
			return []
		#print(self.word_vectors[0])
		#print(word_vec.reshape(1,-1))


		dst = euclidean_distances(normalize(self.word_vectors, axis=0),word_vec.reshape(1,-1))



		word_ids = np.argsort(np.transpose(dst))[0]
		#print(np.transpose(dst).shape, word_ids)
		return [(self.inverse_dictionary[x], dst[x][0]) for x in word_ids[:number]
		        if x in self.inverse_dictionary]



	def is_similar_to(self, word ="", count = 10, method="cos", silent=False):
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
		   self.dictionary[word]
		except:
			print("Word not in context")
			return

		i = 0
		results = []
		import heapq
		for key in self.dictionary.keys():
			if key != word:
				sim = self.get_similarity(word, key, method=method)
				heapq.heappush(results, (sim, key))
			if len(results) > count:
				heapq.heappop(results)

		if not silent:
			for x in results:
				print(x[1],":\t",x[0])

		return results

def main():

	try:
		if sys.argv[1] == 'wordnet':
			if sys.argv[2] :
				wv = Wevis()
				wv.load_model(filepath=sys.argv[2])
				wv.info()
				wv.wordnet("man", method="tsne", num= 10)
				#print(wv.glove_similarity_query("man",10))
	except Exception as e:
		print("Errors", type(e), e.__traceback__)
		traceback.print_exc(file=sys.stdout)


if __name__ == "__main__":
	main()