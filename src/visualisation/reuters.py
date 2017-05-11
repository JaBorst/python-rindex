
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

import pickle
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import sent_tokenize
from helpers import printProgress



directory = "/home/jb/git/reuters-21578-json-master/data/full/"
tmpdir = "tmp/"
model = "reuters.model"

translate_table = dict((ord(char), None) for char in string.punctuation)
stop = set(stopwords.words('english'))

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

	r = RIModel.RIModel(100,5)

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

					if article.get('body') and article.get('id') and article.get('topics') and article.get('places'):
						#Filter TOPICS
						if len(article.get('topics')) == 1 and (set(["grain", "trade","interest","livestock"]) & set(article.get('topics'))):
							places.append(article.get('places')[0])
							topics.append(list(set(["grain", "trade","interest","livestock"]) & set(article.get('topics')))[0])
							r.add_unit(unit=article['id'], context=tokens(article['body']))
							#break
			#break

			print("done")
		else:
			continue

	r.is_similar_to(word="6006")

	r.write_model_to_file(model)
	with open("reuters.places","wb") as placesOutput:
		pickle.dump(places, placesOutput)
	with open("reuters.topics","wb") as topicsOutput:
		pickle.dump(topics, topicsOutput)
	print(set(topics))

def main():
	createReutersModel()

if __name__ == "__main__":
	main()