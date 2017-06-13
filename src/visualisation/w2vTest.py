from gensim.models import word2vec

import logging
from nltk.tokenize import sent_tokenize
from nltk import word_tokenize
import string
from nltk.corpus import stopwords


def main():
	logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.WARN)



	with open("src/Else.txt") as inputFile:
		text = inputFile.read()

	sent_tokenize_list = sent_tokenize(text=text)

	translate_table = dict((ord(char), None) for char in string.punctuation)
	stop = set(stopwords.words('german'))

	sent_tokenize_list_tokens = []

	for sent in sent_tokenize_list:
		tokens = [i.lower() for i in word_tokenize(sent.translate(translate_table)) if i.lower() not in stop and i.isalpha()]
		sent_tokenize_list_tokens.append(tokens)

	print(sent_tokenize_list_tokens[:10])

	model = word2vec.Word2Vec(sent_tokenize_list_tokens, size=200)

	print(model.most_similar(["gulden"], topn=10))


if __name__ == "__main__":
	main()