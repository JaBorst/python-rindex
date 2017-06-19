
import string
import sys
sys.path.append('../')


from docutils.nodes import topic
from  rindex import *
from tsne.tsne import *
import numpy as np
import pylab as Plot
import json
import os
import sys
from nltk.stem import WordNetLemmatizer
import pickle
from nltk import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import sent_tokenize
from helpers import printProgress
import nltk

from rivis import rivis

directory = "/home/jb/git/reuters/full/"
tmpdir = "../models/"
model = "reuters.model"

translate_table = dict((ord(char), None) for char in string.punctuation)
stop = set(stopwords.words('english'))
wl = WordNetLemmatizer()


def tokenizeCorpus(corpus=""):
	global tagger
	translate_table = dict((ord(char), None) for char in string.punctuation)
	sentences = nltk.sent_tokenize(corpus)  # tokenize sentences

	nouns = []  # empty to array to hold all nouns

	for sentence in sentences:

		for line in pos_tag(sentence):
			word=line[0]
			pos = line[1]
			#print(word, pos)
			if (pos == 'NN' or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS'):
				nouns.append(wl.lemmatize(word,"n"))
	return nouns


def tokens(body=""):
	#print (body)
	sent_tokenize_list = sent_tokenize(text=body)
	tokenlist = []
	for sent in sent_tokenize_list:
		tokenlist += [i.lower() for i in word_tokenize(sent.translate(translate_table)) if
		          i.lower() not in stop and i.isalpha()]
	#print("Tokenlist= ",tokenlist)
	return tokenlist

def createReutersModel():

	r = RIModel.RIModel(1000,10)

	numFiles = len(os.listdir(directory))
	j=0
	print("Files found: ",numFiles)

	places = []
	topics = []

	for filename in os.listdir(directory):
		j+=1
		if j== 23:
			break
		if filename.endswith(".json"):
			file = os.path.join(directory, filename)

			with open(file) as data_file:
				data = json.load(data_file)

				i = 0
				numEntries = len(data)
				printProgress(i, numEntries,
				              prefix='File %i/%i Progress files: ' % (j,numFiles), suffix='Complete: %s' %file, barLength=50)

				for article in data:
					i += 1
					if i % 100 == 0:
						printProgress(i, numEntries,
					              prefix='File %i/%i Progress files: ' % (j, numFiles), suffix='Complete: %s' %file, barLength=50)

					if article.get('body') and article.get('id') and article.get('topics'):
						#Filter TOPICS
						if len(article.get('topics')) == 1:
							places.append(article.get('places')[0])
							topics.append(article.get('topics')[0])
							r.add_unit(unit=article['id'], context=tokenizeCorpus(article['body']))
			#				break
			#break

			print("done")
		else:
			continue

	print("\n")
	riv = rivis.Rivis(r)
	riv.tsne2_calc()
	riv.set_visualisation_dict(name="labels", data=topics)
	riv.set_visualisation_dict(name="places", data=places)
	riv.set_title("ReutersDataSet")
	riv.info()
	with open("../model/reuters.riv", "wb") as dump:
		pickle.dump(riv, dump)


def main():
	createReutersModel()

if __name__ == "__main__":
	main()