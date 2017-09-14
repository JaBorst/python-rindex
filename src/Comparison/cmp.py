import sys
import traceback
sys.path.append("../")
from gensim.models.word2vec import Text8Corpus
import itertools
from nltk.corpus import brown
from  nltk.corpus import stopwords
from math import log2 as lg
from math import ceil
from math import fabs
import pickle
from nltk.tokenize import word_tokenize
import gzip
import os
swords = stopwords.words('german')

data = []
windowsize = 6
target_size = 100

word_representation_dict = {}
filename="/run/media/jb/Black/Corpora/Models/asv_wortschatz_%s.model"


class MySentences(object):
	def __init__(self, fname):
		self.filename = fname
		self.current_line = 0

	def __iter__(self):
		for line in gzip.open(self.filename):
			self.current_line= self.current_line + 1
			if self.current_line % 100 == 0:
				sys.stdout.write('\r%s' % (self.current_line))
			# yield word_tokenize(line.decode('utf-8'))
			yield line.decode('utf-8').split(" ")

	def get_count(self):
		return self.current_line

	def __len__(self):
		return 9000000

def genGloveModel():
	from glove import Glove, Corpus

	global data
	global windowsize
	global target_size
	global word_representation_dict

	print("Loading Wordlist...")
	with open('/run/media/jb/Black/Corpora/Models/w2v.words', 'rb') as wordlistInput:
		word_list = pickle.load(wordlistInput)

	# Fit the co-occurrence matrix using a sliding window of 10 words.
	corpus = Corpus()
	corpus.fit(data, window=windowsize)
	glove = Glove(no_components=target_size, learning_rate=0.05)
	print("\n")
	#print(corpus.matrix)
	#print(corpus.dictionary)
	glove.fit(corpus.matrix, epochs=30, no_threads=4, verbose=True)
	glove.add_dictionary(corpus.dictionary)

	for key in glove.dictionary.keys():
		if key in word_list:
			word_representation_dict[key] = glove.word_vectors[glove.dictionary[key]]

	with open(filename % ("glove_con"),"wb") as of:
		pickle.dump(word_representation_dict, of)

	print(glove.most_similar('man'))

def genRIModel():
	global target_size
	global data
	global windowsize

	from rindex.RIModel import RIModel
	print("Creating RIndex Model")
	no_sentence = len(data)
	ri = RIModel(dim=max(ceil(10*lg(no_sentence)),150),k=ceil(lg(lg(no_sentence))))
	print("Initial Dimensions of the Model:",ri.dim,ri.k)
	for sent_index,sent in enumerate(data):
		# sys.stdout.write('\r%s/%s' % (sent_index,no_sentence))

		sent_nostop = [w.replace("'s",'') for w in sent if len(w) > 2 and w.lower() not in swords]

		for index, word  in enumerate(sent_nostop):

			lower_index = max(0, int(index-windowsize/2))
			upper_index = min(len(sent_nostop), int(index+windowsize/2))
			context = [w for w in sent_nostop[lower_index: upper_index] ]
			weights = []

			for i in range(lower_index,upper_index):

				dist=fabs(i-index)
				if dist != 0:
					weights.append(1-lg(dist)/lg(windowsize/2+1))
				else:
					weights.append(0)


			ri.add_unit(unit=word, context=context, weights=weights)
	print("\n")



	ri.reduce_dimensions(method="truncated_svd", target_size=target_size)


	word_representation_dict = {}
	for key in ri.ContextVectors.keys():
		# print(type(ri.ContextVectors[key]))
		word_representation_dict[key] = ri.ContextVectors[key]

	with open(filename % ("rindex_NoDecap_unnormed"), "wb") as of:
		pickle.dump(word_representation_dict, of)
	word_representation_dict = {}

	ri.truncate(treshold = 0.0)
	for key in ri.ContextVectors.keys():
		#print(type(ri.ContextVectors[key]))
		word_representation_dict[key]  = ri.ContextVectors[key]

	with open(filename % ("rindex_NoDecap_normed"),"wb") as of:
		pickle.dump(word_representation_dict, of)
	word_representation_dict = {}

	ri.truncate(treshold=0.1)
	for key in ri.ContextVectors.keys():
		#print(type(ri.ContextVectors[key]))
		word_representation_dict[key]  = ri.ContextVectors[key]

	with open(filename % ("rindex_NoDecap_truncated01"),"wb") as of:
		pickle.dump(word_representation_dict, of)




def genRIModel_confined():
	global target_size
	global data
	global windowsize

	from rindex.RIModel import RIModel
	print("Creating RIndex Model")
	no_sentence = len(data)
	print("Loading Wordlist...")
	with open('/run/media/jb/Black/Corpora/Models/w2v.words', 'rb') as wordlistInput:
		word_list = pickle.load(wordlistInput)

	if len(word_list) == 0:
		print("No wordlist loaded...Exit")
		exit()



	ri = RIModel(dim=max(ceil(10*lg(no_sentence)),150),k=ceil(lg(lg(no_sentence))))
	print("Initial Dimensions of the Model:",ri.dim,ri.k)



	for sent_index,sent in enumerate(data):
		# sys.stdout.write('\r%s/%s' % (sent_index,no_sentence))

		sent_nostop = [w.replace("'s",'') for w in sent if len(w) > 2 and w.lower() not in swords]

		for index, word  in enumerate(sent_nostop):

			if word in word_list:
				lower_index = max(0, int(index-windowsize/2))
				upper_index = min(len(sent_nostop), int(index+windowsize/2))
				context = [w for w in sent_nostop[lower_index: upper_index]]
				weights = [ 1-lg(fabs(i-index))/lg(windowsize/2+1) if i != index else 0 for i in range(lower_index,upper_index) ]
				ri.add_unit(unit=word, context=context, weights=weights)
	print("\n")



	ri.reduce_dimensions(method="truncated_svd", target_size=target_size)
	with open(filename % ("rindex_NoDecap_unnormed"), "wb") as of:
		pickle.dump(ri.ContextVectors, of)


	ri.truncate(threshold = 0.0)
	with open(filename % ("rindex_NoDecap_normed"),"wb") as of:
		pickle.dump(ri.ContextVectors, of)
	word_representation_dict = {}

	ri.truncate(threshold=0.1)
	with open(filename % ("rindex_NoDecap_truncated01"),"wb") as of:
		pickle.dump(ri.ContextVectors, of)

def genW2VModel():
	global data
	global windowsize
	global target_size
	from gensim.models import Word2Vec
	model = Word2Vec(data, size=target_size, window=windowsize, min_count=5, workers=4)

	for key in model.wv.vocab.keys():
		#print(type(model.wv.word_vec(key)))
		word_representation_dict[key] = model.wv.word_vec(key)
	with open(filename % ("w2v"),"wb") as of:
		pickle.dump(word_representation_dict, of)

def main():

	try:
		if sys.argv[1] == "gen":
			global data
			if sys.argv[3]:
				if sys.argv[3] == "brown":
					data = brown.sents(categories=brown.categories()) # brown.categories())
				elif "/" in sys.argv[3]:
					print("Assuming this is a path to a gzipped file...")

					data = MySentences(sys.argv[3])
					# with gzip.open(sys.argv[3],"r") as input:
					# 	for line in input:
					# 		#print(line.decode('utf-8'))
					# 		data.append(word_tokenize(line.decode('utf-8')))
					# 		#print (data)

			if sys.argv[2] == "glove":
				genGloveModel()
			if sys.argv[2] == "rindex":
				genRIModel()
			if sys.argv[2] == "rindex_con":
				genRIModel_confined()

			if sys.argv[2] == "w2v":
				genW2VModel()
			if sys.argv[2] == "all":
				genGloveModel()
				data.current_line=0
				genRIModel()
				data.current_line=0
				genW2VModel()

	except Exception as e:
		print("Errors", type(e), e.__traceback__)
		traceback.print_exc(file=sys.stdout)

	else:
		print(sys.argv[1:], "successfully executed")
	finally:
		print("Closing.")

if __name__ == "__main__":
	main()
