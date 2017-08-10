#
# __filename__: t_helper.py
#
# __description__:
#
# __remark__:
#
# Created by Tobias Wenzel on 10-08-17.
# Copyright (c) 2017 Tobias Wenzel

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
import time
import datetime

line_length= 60
colon_line = ":"*line_length


def clean_word_seq(context = []):
    """
    # some trimming
    :param context: 
    :return: 
    """
    stemmer = SnowballStemmer("english",ignore_stopwords=True)
    #not in stop
    # stop-words are for suckers...
    if len(context) > 1:
        return [word.lower() for word in context if word.isalpha() and len(word) > 1]
    elif len(context) == 0:
        return
    elif len(context) == 1:
        if context[0].isalpha() and context[0]:
            return context[0].lower()
        else:
            return


def get_paths_of_files(path="", filetype=".txt"):
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


def log_model(method):
    """
    called with decorator in build_model
    :param method:
    :return:
    """
    def write_to_log(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        with open("/home/tobias/Dokumente/models/model.log", "a") as fout:
            fout.write(colon_line+"\n")
            fout.write(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')+"\n")
            fout.write('%r \n  %r  \n\n %2.2f sec\n' % \
                  (method.__name__,  kw, te - ts))
            fout.write(colon_line+"\n")

        return result

    return write_to_log


def split_text_file(filename="", chunks=1):
    """

    :param filename:
    :param chunks:
    :return:
    """
    with open(filename, 'r', encoding='utf-8') as fin:
        text = fin.readlines()
    lines_in_new_file = int(len(text) / chunks)
    index = 0
    for i in range(chunks):
        n_filename= os.path.split(filename)[0]+"/"+str(i)
        with open(n_filename, 'a') as fout:
            for j in range(lines_in_new_file):
                fout.write(text[index]+"\n")
                index+=1

"""
    Testing
"""
def sentence_mean_sd(filename=""):
    """
    return mean and sd of sentences
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
    can be used to split the state of union to create ri-model
    for each president.
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

# @depr
# def search_in_matrix(matrix, keys=[], word=""):
#     """
#     ...shouldn't be necessary by now
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

if __name__ == '__main__':
    print(sentence_mean_sd())