
import string
import sys

import pickle

sys.path.append('../')

from  rindex import *
from tsne.tsne import *
from rivis import rivis

from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from helpers import printProgress

import nltk
import gzip
import re
pattern_noun = re.compile("^NN")
pattern_verb = re.compile("^VB")
pattern_adj = re.compile("^JJ")




directory = "/home/jb/git/reuters-21578-json/data/full/"
tmpdir = "../models/"
model = "reuters.model"

translate_table = dict((ord(char), None) for char in string.punctuation)
stop = set(stopwords.words('english'))
wl = WordNetLemmatizer()


def tokenizeCorpus(corpus=""):
	global tagger
	translate_table = dict((ord(char), None) for char in string.punctuation)
	sentences = nltk.sent_tokenize(corpus)  # tokenize sentences

	words = []  # empty to array to hold all nouns

	for sentence in sentences:

		for line in pos_tag(word_tokenize(sentence.translate(translate_table))):
			word=line[0]
			pos = line[1]
			#print(word, pos)
			if (pattern_noun.match(pos) or pattern_adj.match(pos) or pattern_verb.match(pos)):
				words.append(wl.lemmatize(word,"n"))

	return words

def tokenizeCorpusWeighted(corpus=""):
	global tagger
	translate_table = dict((ord(char), None) for char in string.punctuation)
	sentences = nltk.sent_tokenize(corpus)  # tokenize sentences

	words = []  # empty to array to hold all nouns
	weights = []

	for sentence in sentences:

		for line in pos_tag(word_tokenize(sentence.translate(translate_table))):
			word=line[0]
			pos = line[1]
			#print(word, pos)
			if pattern_noun.match(pos):
				words.append(wl.lemmatize(word, "n"))
				weights.append(1.0)
			elif pattern_adj.match(pos):
				words.append(wl.lemmatize(word, "n"))
				weights.append(1.0)
			elif pattern_verb.match(pos):
				words.append(wl.lemmatize(word, "n"))
				weights.append(0.2)

	return words,weights


def tokens(body=""):
	#print (body)
	sent_tokenize_list = sent_tokenize(text=body)
	tokenlist = []
	for sent in sent_tokenize_list:
		tokenlist += [i.lower() for i in word_tokenize(sent.translate(translate_table)) if
		          i.lower() not in stop and i.isalpha()]
	#print("Tokenlist= ",tokenlist)
	return tokenlist



def createWikiPlotsModel():
	titleFile = "/home/jb/git/wikiplots/titles1000.gz"
	storyFile = "/home/jb/git/wikiplots/plots.gz"


	ri = RIModel.RIModel(1000,10)
	riv = rivis.Rivis()

	with gzip.open(titleFile, 'rb') as f, gzip.open(storyFile, "rb") as sf:
		plots = sf.read().decode("utf-8").split("<EOS>")
		titles = f.readlines()


		i=0
		numEntries = len(titles)
		numPlots = len(plots)
		print(numEntries, numPlots)

		for (title, plot) in zip(titles, plots):
			i = i + 1
			printProgress(i, numEntries,
						  prefix='Movie Progress : ', suffix='Completed %i/%i plots' % (i, numEntries), barLength=50)

			title = title.decode("utf-8")
			#print(title, plot)
			t = tokenizeCorpus(plot)
			#print(t)
			words, weights  = tokenizeCorpusWeighted(plot)
			#print(weights)
			ri.add_unit(unit=title, context=words, weights = weights)


		print("\n")
		riv = rivis.Rivis(ri)
		riv.tsne2_calc()
		riv.set_visualisation_dict(name = "labels", data = titles)
		riv.set_title("movieplots")
		riv.info()
		with open("movieplots.riv","wb") as dump:
			pickle.dump(riv,dump)


def main():
	print("Starting...")
	createWikiPlotsModel()

if __name__ == "__main__":
	main()
