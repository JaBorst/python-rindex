#
# __filename__: clustering.py
#
# __description__:
# __remark__: @deprecated please don't use!

import string
import collections

from nltk import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from pprint import pprint

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt



def process_text(text, stem=True):
    """ Tokenize text and stem words removing punctuation """
    #text = text.translate(None, string.punctuation)
    tokens = word_tokenize(text)
    if stem:
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(t) for t in tokens]

    return tokens



if __name__ == "__main__":
    articles = ""
    with open("/home/tobias/Dokumente/partei-test.txt","r") as fin:
        articles= fin.read()


    pipeline = Pipeline([
        ('vect', CountVectorizer()), # ?
        ('tfidf', TfidfTransformer()),
    ])
    tfidf_model = pipeline.fit_transform(process_text(articles))

    data2D = TruncatedSVD(n_components=2, n_iter=20, random_state=42).fit_transform(tfidf_model)

    plt.scatter(data2D[:, 0], data2D[:, 1])
    plt.show()
    #
    # clusters = cluster_texts([articles,""], 2)
    # pprint(dict(clusters))



    #def cluster_texts(texts, clusters=3):

    # """ Transform texts to Tf-Idf coordinates and cluster texts using K-Means """
    #     vectorizer = TfidfVectorizer(tokenizer=process_text,
    #                                  # stop_words=stopwords.words('english'),
    #                                  # max_df=.9,
    #                                  # min_df=0.01,
    #                                  lowercase=True)
    #
    #     tfidf_model = vectorizer.fit_transform(texts)
    #     print(tfidf_model)
    #     km_model = KMeans(n_clusters=clusters)
    #     km_model.fit(tfidf_model)
    #
    #     clustering = collections.defaultdict(list)
    #
    #     for idx, label in enumerate(km_model.labels_):
    #         clustering[label].append(idx)
    #
    #     return clustering