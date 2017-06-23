import sys
sys.path.append('../')
#from rindex import *
import os
from nltk.corpus import stopwords
stop = set(stopwords.words('english'))
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize as w_tokenize
from nltk.tokenize import sent_tokenize as s_tokenize
import numpy as np

def clean_word_seq(context = []):
    """
    # some trimming
    :param context: 
    :return: 
    """
    stemmer = SnowballStemmer("english",ignore_stopwords=False)
    #not in stop
    # stop-words are for suckers...
    if len(context) > 1:
        return [word.lower() for word in context if word.isalpha() and len(word)>1]
    elif len(context) == 0:
        return
    elif len(context) == 1:
        if context[0].isalpha():
            return context[0].lower()
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
#
# def merge_dicts(d1={}, d2={}):
#     """
#     as the name suggests: if we have more than one dict/model
#     this is an incremental function to merge them
#     :param d1:
#     :param d2:
#     :return:
#     """
#     nn = {}
#     for key in d1.keys():
#         if key in d2.keys():
#             nn[key] = d1[key]+d2[key]
#         else:
#             nn[key] = d1[key]
#     for key in d2.keys():
#         if key not in d1.keys():
#             nn[key] = d2[key]
#     rim = RIModel(dim=1500,k=3)
#
#     rim.ContextVectors = nn
#     rim.write_model_to_file("accu")
#     print("finished merging")



# def normalize_matrix(matrix):
#     from sklearn.preprocessing import normalize
#     return normalize(matrix, axis=1, norm='l2',copy=False)


# def search_in_matrix(matrix, keys=[], word=""):
#     """
#
#     :param matrix:
#     :param keys:
#     :param word:
#     :return:
#     """
#     from scipy import spatial
#     word_iv = matrix[keys.index(word)]
#
#     if sp.issparse(word_iv):
#         word_iv = word_iv.toarray()
#
#     max_d = 0
#     max_key = ""
#     for key in keys:
#         if key != word:
#             d = 1 - spatial.distance.cosine(word_iv, matrix[keys.index(key)])  # .toarray())
#             if d > max_d:
#                 max_d = d
#                 max_key = key
#     print(max_d, max_key)


def sentence_mean_sd(filename=""):
    """

    :param filename:
    :return:
    """
    filename = "/home/tobias/Dokumente/testdata/stateofunion.txt"

    with open(filename, 'r') as fin:
        text = fin.read()
    sents = s_tokenize(text)
    num_words_of_sents = np.zeros(len(sents))
    for i in range(len(sents)):
        num_words_of_sents[i] = len(w_tokenize(sents[i]))
    print(num_words_of_sents)
    return np.mean(num_words_of_sents), np.std(num_words_of_sents)

def split_state_of_union():
    """

    :return:
    """
    import re

    exp = "\*\*\*"
    filename = "/home/tobias/Dokumente/testdata/stateofunion.txt"

    with open(filename, 'r') as fin:
        text = []
        target_file = ""
        collect = False
        president = ""
        for line in fin:
            if re.match(exp, line):
                if text and president:
                    with open(target_file, 'a') as fout:
                        fout.writelines(text)
                text = []
                fin.readline()
                fin.readline()
                president = fin.readline().strip()
                text.append(president)
                target_file = "/home/tobias/Dokumente/testdata/presidents/" \
                              + president + ".txt"

            else:
                text.append(line)


def get_n_sents(n=100, filename=""):
    """
    to get text which is +- equal in size (or at least number of sents)
    :param n:
    :param filename:
    :return:
    """
    filename = "/home/tobias/Dokumente/testdata/stateofunion.txt"
    text = ""
    with open(filename, 'r') as fin:
        text = fin.read()
    sents = s_tokenize(text)
    indices = np.random.permutation(range(len(sents)))
    return [sents[indices[i]] for i in range(n)]

if __name__ == '__main__':
    print(sentence_mean_sd())