import sys
import traceback
sys.path.append("../")
from gensim.models.word2vec import Text8Corpus
import itertools
from nltk.corpus import brown
from  nltk.corpus import stopwords
from math import log2 as lg
from math import ceil

swords = stopwords.words('english')

data = []
windowsize= 10



def genGloveModel():
	from glove import Glove, Corpus
	import gzip
	global data
	global windowsize

	#data = (str(line).lower().split(' ') for line
	#		in gzip.open('/home/jb/git/wikiplots/plots1000.gz', 'r'))

	for d in data:
		print(d)
		break


	# Fit the co-occurrence matrix using a sliding window of 10 words.
	corpus = Corpus()
	corpus.fit(data, window=windowsize)
	glove = Glove(no_components=100, learning_rate=0.05)
	#print(corpus.matrix)
	#print(corpus.dictionary)
	glove.fit(corpus.matrix, epochs=30, no_threads=4, verbose=True)
	glove.add_dictionary(corpus.dictionary)
	print(glove.most_similar('man'))

def genRIModel():
	from rindex.RIModel import RIModel

	no_sentence = len(data)
	ri = RIModel(dim=max(ceil(10*lg(no_sentence)),150),k=ceil(lg(lg(no_sentence))))
	print(ri.dim,ri.k)
	for i,sent in enumerate(data):
		sys.stdout.write('\r%s/%s' % (i,no_sentence))
		#print(sent)
		sent = [w for w in sent if len(w) > 2 and w.lower() not in swords]
		for index, word  in enumerate(sent):
			#print(index, word.lower())
			context = [w for w in sent[max(0, index-windowsize): min(len(sent)-1, index+windowsize)] if w != word]
			ri.add_unit(unit=word.lower(), context=context)

	ri.reduce_dimensions(method="truncated_svd", target_size=100)
	print(ri.is_similar_to(word="man",method="jaccard"))


def main():

	try:
		if sys.argv[1] == "gen":
			global data
			data = brown.sents(categories=['news'])[:100]#, 'editorial', 'reviews', 'adventure'])  # brown.categories())

			if sys.argv[2] == "glove":
				genGloveModel()
			if sys.argv[2] == "rindex":
				genRIModel()

	except Exception as e:
		print("Errors", type(e), e.__traceback__)
		traceback.print_exc(file=sys.stdout)

	else:
		print(print(sys.argv), "successfully executed")
	finally:
		print("Closing.")

if __name__ == "__main__":
	main()