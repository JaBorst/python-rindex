import sys
sys.path.append('../')
from rindex import *
import os
from nltk.corpus import stopwords
stop = set(stopwords.words('english'))
from nltk.stem.snowball import SnowballStemmer


def clean_word_seq(context = []):
    """
    # some trimming
    :param context: 
    :return: 
    """
    #stemmer = SnowballStemmer("english")

    if len(context) > 1:
        return [word.lower() for word in context if word.lower() not in stop and word.isalpha()]
    elif len(context) == 0:
        return
    elif len(context) == 1:
        if context[0].lower() not in stop and context[0].isalpha():
            return stemmer.stem(context[0].lower())
        else:
            return


def get_paths_of_files(path = "", filetype=".txt"):
    """

    :param path: 
    :return: 
    """
    list_of_files = []
    i = 0
    for (dirpath, dirnames, filenames) in os.walk(path):
        for filename in filenames:
            if filename.endswith(filetype):
                list_of_files.append(os.sep.join([dirpath, filename]))
    return list_of_files

def merge_dicts(d1={}, d2={}):
    """
    as the name suggests: if we have more than one dict/model
    this is an incremental function to merge them
    :param d1:
    :param d2:
    :return:
    """
    nn = {}
    for key in d1.keys():
        if key in d2.keys():
            nn[key] = d1[key]+d2[key]
        else:
            nn[key] = d1[key]
    for key in d2.keys():
        if key not in d1.keys():
            nn[key] = d2[key]
    rim = RIModel(dim=1500,k=3)

    rim.ContextVectors = nn
    rim.write_model_to_file("accu")
    print("finished merging")



def normalize_matrix(matrix):
    from sklearn.preprocessing import normalize
    return normalize(matrix, axis=1, norm='l2')


def search_in_matrix(matrix, keys=[], word=""):
    """

    :param matrix: 
    :param keys: 
    :param word: 
    :return: 
    """
    from scipy import spatial
    word_iv = matrix[keys.index(word)]

    if sp.issparse(word_iv):
        word_iv = word_iv.toarray()

    max_d = 0
    max_key = ""
    for key in keys:
        if key != word:
            d = 1 - spatial.distance.cosine(word_iv, matrix[keys.index(key)])  # .toarray())
            if d > max_d:
                max_d = d
                max_key = key
    print(max_d, max_key)