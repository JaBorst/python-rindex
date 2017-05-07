from  rindex import *
from tsne.tsne import *
import numpy as np
import pylab as Plot
import json
import os


from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import sent_tokenize
from helpers import printProgress


import string

directory = "/home/jb/git/reuters-21578-json-master/data/full/"
model = "reuters.model"


def createReutersModel():

	r = RIModel.RIModel(10000,5)

	numFiles = len(os.listdir(directory))
	j=0
	print("Files found: ",numFiles)

	for filename in os.listdir(directory):
		j+=1
		if j== 10:
			break
		if filename.endswith(".json"):
			file = os.path.join(directory, filename)

			with open(file) as data_file:
				data = json.load(data_file)

				i = 0
				numEntries = len(data)
				printProgress(i, numEntries,
				              prefix='File %i/%i Progress files: ' % (j,numFiles), suffix='Complete', barLength=50)

				for article in data:
					i += 1
					if i % 100 == 0:
						printProgress(i, numEntries,
					              prefix='File %i/%i Progress files: ' % (j, numFiles), suffix='Complete', barLength=50)

					if article.get('body') and article.get('id') and article.get('topics'):
						sent_tokenize_list = sent_tokenize(text=article['body'])
						translate_table = dict((ord(char), None) for char in string.punctuation)
						stop = set(stopwords.words('english'))
						for sent in sent_tokenize_list:
							tokens = [i.lower() for i in word_tokenize(sent.translate(translate_table)) if
							          i.lower() not in stop and i.isalpha()]
							r.addUnit(unit=article['id'], context=tokens)

			print("done")
		else:
			continue

	r.writeModelToFile(model)

def main():
	createReutersModel()

if __name__ == "__main__":
	main()