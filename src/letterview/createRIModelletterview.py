import sys
sys.path.append('../')

import treetaggerwrapper

from  rindex import *
import sqlite3
from nltk import word_tokenize, FreqDist
from nltk.corpus import stopwords
from nltk.tag import StanfordPOSTagger
import string
import nltk
from collections import Counter
import re

tagger = treetaggerwrapper.TreeTagger(TAGLANG='de')

def tokenizeCorpus(corpus=""):
	global tagger
	translate_table = dict((ord(char), None) for char in string.punctuation)
	sentences = nltk.sent_tokenize(corpus)  # tokenize sentences

	nouns = []  # empty to array to hold all nouns

	for sentence in sentences:

		for line in tagger.tag_text(sentence):
			word=line.split("\t")[0]
			pos = line.split("\t")[1]
			print(word, pos)
			if (pos == 'NN' or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS'):
				nouns.append(word)
	return nouns



def main():
	ri = RIModel.RIModel(dim=100, k=10)

	conn = sqlite3.connect('/home/jb/git/letterview/app/misc/example.db')
	c = conn.cursor()
	c.execute("SELECT id,content from letters;")


	statsconn = sqlite3.connect('/home/jb/git/letterview/app/backend/lingstats.db')
	statsc = statsconn.cursor()

	for (id, letter) in c:
		print(id)

		words = tokenizeCorpus(re.sub(' +',' ',letter.replace('\n','').replace('\r','')))
		for w in words:
			sql = "SELECT tf.freq *1.0/df.freq  from termfreq tf JOIN dokfreq df ON (tf.word=df.word) where tf.docID = %i and tf.word = '%s'" % (id,w.lower())
			print(sql)
			res = statsc.execute(sql)
			data = res.fetchone()
			if data == None:
				print("No Data Found")
			else:
				tfidf = data[0]
				print(tfidf)
				ri.add_unit(unit=str(id), context=[w], weights = [tfidf])

		break


if __name__ == "__main__":
	main()