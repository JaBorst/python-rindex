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



swords = stopwords.words('english')

data = []
windowsize = 10
target_size = 100

word_representation_dict = {}
filename="brown_all_%s.model"



def genGloveModel():
	from glove import Glove, Corpus
	import gzip
	global data
	global windowsize
	global target_size
	global word_representation_dict

	#data = (str(line).lower().split(' ') for line
	#		in gzip.open('/home/jb/git/wikiplots/plots1000.gz', 'r'))

	for d in data:
		print(d)
		break


	# Fit the co-occurrence matrix using a sliding window of 10 words.
	corpus = Corpus()
	corpus.fit(data, window=windowsize)
	glove = Glove(no_components=target_size, learning_rate=0.05)
	#print(corpus.matrix)
	#print(corpus.dictionary)
	glove.fit(corpus.matrix, epochs=30, no_threads=4, verbose=True)
	glove.add_dictionary(corpus.dictionary)

	for key in glove.dictionary.keys():
		word_representation_dict[key] = glove.word_vectors[glove.dictionary[key]]

	with open(filename % ("glove"),"wb") as of:
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
		sys.stdout.write('\r%s/%s' % (sent_index,no_sentence))

		sent_nostop = [w.replace("'s",'') for w in sent if len(w) > 2 and w.lower() not in swords]

		for index, word  in enumerate(sent_nostop):

			lower_index = max(0, int(index-windowsize/2))
			upper_index = min(len(sent_nostop), int(index+windowsize/2))
			context = [w for w in sent_nostop[lower_index: upper_index]]
			weights = []

			for i in range(lower_index,upper_index):

				dist=fabs(i-index)
				if dist != 0:
					weights.append(1-lg(dist)/lg(windowsize/2+1))
				else:
					weights.append(0)


			ri.add_unit(unit=word.lower(), context=context, weights=weights)

	ri.reduce_dimensions(method="truncated_svd", target_size=target_size)

	for key in ri.ContextVectors.keys():
		#print(type(ri.ContextVectors[key]))
		word_representation_dict[key]  = ri.ContextVectors[key]

	with open(filename % ("rindex"),"wb") as of:
		pickle.dump(word_representation_dict, of)


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
			data = brown.sents(categories=brown.categories()) # brown.categories())

			if sys.argv[2] == "glove":
				genGloveModel()
			if sys.argv[2] == "rindex":
				genRIModel()
			if sys.argv[2] == "w2v":
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