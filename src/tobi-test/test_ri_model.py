#
# __filename__: test_ri_model.py
#
# __description__: This script can be used to compare cvs (words
# or documents) in RI-Models which have been created in build_model.py.
#
# __remark__: Mind the different methods to add context vectors!
#
# Created by Tobias Wenzel
# Copyright (c) 2017 Tobias Wenzel


import sys
sys.path.append('../')

import re
import os
import pickle
import time
import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import KDTree

import matplotlib.pyplot as plt

from pathlib import Path
from os.path import basename
from t_helper import get_paths_of_files

from rindex import *

# used for nice printings
line_length= 60
dotted_line = "-"*line_length
line = "_"*line_length
colon_line = ":"*line_length
quarter_space_line = " "*int(line_length/4)



def search_tree_for_similar(kdt, keys, method="query", n=10, word=""):
    """

    :param kdt: 
    :param keys: 
    :param method: query
    :param n: number of similar words
    :param word: 
    :return: 
    """
    some_element = np.array(kdt.data[keys.index(word)])
    if method == "query":
        dist, ind = kdt.query(some_element.reshape(1, -1), k=n)
    for i, d in zip(ind[0], dist[0]):
        print("{0}\t{1: >#016.4f}".format(keys[i], d))



# """
#     TESTING
#   -> tranlating
#
#
# """
#
#
# import heapq
# from scipy import spatial
#
#vec_is_similar_to(rmi, vec=rmi2.ContextVectors['mann'].toarray(), count=10, silent=False )
# def vec_is_similar_to(rmi, vec, count=10, silent=False):
#
#
#     results = []
#     for key in rmi.ContextVectors.keys():
#         sim = 1 - spatial.distance.cosine(vec, rmi.ContextVectors[key].toarray())
#
#         heapq.heappush(results, (sim, key))
#         if len(results) > count:
#             heapq.heappop(results)
#     results = heapq.nlargest(count, results)
#     if not silent:
#         for x in results:
#             print(x[1], ":\t", x[0])
#
#     return results
#
# """
#     TESTING ENDE
#
# """




def visualize_vecs(rmi, vec_names=[]):
    if len(vec_names) == 0:
        return
    ivs = []
    for name in vec_names:
        ivs.append(rmi.ContextVectors[name].toarray())
    x = [i for i in range(ivs[0].shape[0])]
    f, axarr = plt.subplots(len(vec_names), sharex=True)
    for i in range(len(vec_names)):
        axarr[i].plot(x, ivs[i])
        axarr[i].set_title(vec_names[i])
        axarr[i].set_ylabel('value')
    axarr[-1].set_xlabel('vec position')
    plt.show()


def main():
    dim = 1500
    k = 3
    rmi = RIModel.RIModel(dim, k)
    context_size = 3
    print("contextsize ist !!!\t",context_size)
    rmi.is_sparse = True
    print(colon_line)
    print(quarter_space_line,"Random Indexing Model")
    print(colon_line)
    mode = input("""\t[1]: search \t 2: visualize\n\t 
    2: load, reduce and build tree\n""") or 0
    reduction = input("\t[1] do reduce \t [0] don't reduce") or 0

    """
        load models
    """
    model_filename = "/home/tobias/Dokumente/models/doc-models/books/books-c4.model"
    rmi.load_model_from_file(model_filename)
    # bei längeren Bücher haut's mir da durch die Decke
    # -> evtl erst reduzieren und dann truncate?
    rmi.truncate(threshold=0.1)
    if int(reduction) == 1:
        rmi.reduce_dimensions(method="tsne", target_size=2)


    # print(rmi.ContextVectors)
    if int(mode) == 0:
        rmi.is_similar_to(word="shakespeare_romeo_and_juliet", count=5, method="cos",silent=False)
        #rmi.most_similar(count=-1,method="jsd",silent=False)

    elif int(mode) == 2:
        vec_names = ['goethe_faust', 'shakespeare_romeo_and_juliet',
                     'goethe_faust', 'shakespeare_henry_the_fifth']
        visualize_vecs(rmi, vec_names)
    elif int(mode) == 3:
        """
            Themen Clustering
        """
        from sklearn.cluster import KMeans
        import collections
        keys, matrix = rmi.to_matrix(to_sparse=True)
        km_model = KMeans(n_clusters=4)
        km_model.fit(matrix)
        clustering = collections.defaultdict(list)
        for idx, label in enumerate(km_model.labels_):
            clustering[label].append(idx)
        for key, value in clustering.items():
            print(key, [keys[idx] for idx in value])

    # if int(mode) == 0 or int(mode)== 2:
         """
             Kd-Tree
         """
    #     tree_filename = "/home/tobias/Dokumente/models/trees/word_sim_3c.tree"
    #     leafs_filename = "/home/tobias/Dokumente/models/trees/word_sim_3c.keys"
    # if int(mode) == 2:
    #     keys, kdt = rmi.to_tree(method="minkowski", leaf_size=50)
    #     with open(tree_filename, 'wb') as output:
    #         pickle.dump(kdt, output)
    #     with open(leafs_filename, 'wb') as output:
    #         pickle.dump(keys, output)
    #
    # elif int(mode) == 0:
    #     with open(tree_filename, 'rb') as inputFile:
    #         kdt = pickle.load(inputFile)
    #     with open(leafs_filename, 'rb') as inputFile:
    #         keys = pickle.load(inputFile)
    #
    # if int(mode) == 0 or int(mode)== 2:
    #     while True:
    #         search_word = input("search for word:\n")
    #
    #         try:
    #             # from nltk.stem.snowball import SnowballStemmer
    #             # search_word = SnowballStemmer("english",ignore_stopwords=False).stem(search_word)
    #             search_tree_for_similar(kdt, keys, method="query", n=7, word=search_word)
    #         except:
    #             print("word not in contextvectors.")
    #         if search_word == "stop":
    #             break


if __name__ == '__main__':
    main()

    # from sklearn.preprocessing import normalize
    # values = normalize(matrix, axis=1, norm='l2', copy=False)
    # for key, i in zip(keys,range(len(keys))):
    #     rmi.ContextVectors[key]=values[i,:].transpose()
    # print(rmi.ContextVectors)

    # """
    # #     tsne
    # """
    # rmi.reduce_dimensions(method="tsne", target_size=2)
    #
    # keys, matrix = rmi.to_matrix()
    # #
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # plt.plot(matrix[:, 0], matrix[:, 1], 'o')
    #
    # for i, x, y in zip(range(len(keys)), matrix[:, 0], matrix[:, 1]):  # <--
    #     ax.annotate('%s' % keys[i], xy=(x, y), textcoords='data')  # <--
    # plt.grid()
    # plt.show()
    # # """

    """
        Vektor-Arithmetik
    """


    # from scipy import spatial
    #
    # word_iv = rmi.ContextVectors['king']-rmi.ContextVectors['man']+rmi.ContextVectors['woman']
    # # king-man ~ royal
    # max_d = 0
    # max_key = ""
    # for key in rmi.ContextVectors.keys():
    #     d = spatial.distance.cosine(word_iv.toarray(), rmi.ContextVectors[key].toarray())  # .toarray())
    #     if max_d<d:
    #         max_d = d
    #         max_key = key
    #         print(max_d, max_key)
    # print(max_d, max_key)


    # with open("/home/tobias/Dokumente/saved_context_vectors/paratest/word_sim.keys", 'rb') as inputFile:
    #     keys = pickle.load(inputFile)
    # with open("/home/tobias/Dokumente/saved_context_vectors/paratest/word_sim.tree", 'rb') as inputFile:
    #     kdt = pickle.load(inputFile)
    #
    #for i in range(10):





    # depr.
    # @depr. see paratest
    # def build_word_sim_model(rmi, path="", context_size=2):
    #     """
    #     walks through all of the files 2006-2015 (eng) to build a
    #     model for word simi.
    #     :param rmi:
    #     :param path:
    #     :param context_size:
    #     :return:
    #     """
    #     files = get_paths_of_files(path, filetype=".txt")
    #     #files = ["/home/tobias/Dokumente/testdata/eng_news_2015_10K-sentences.txt"]
    #     for file in files:
    #         with open(file, 'r', encoding="utf-8") as fin:
    #             sents = fin.readlines()
    #         print(file)
    #         size = len(sents)
    #         for i,sent in zip(range(len(sents)),sents):
    #             if i % 100 == 0:
    #                 print("\r%f %%" % (100 * i / size), end="")
    #             sent = clean_word_seq(w_tokenize(sent))
    #             for j in range(len(sent)):
    #                 context = []
    #                 for k in range(j, j + context_size):
    #                     try:
    #                         context.append(sent[k])
    #                     except:
    #                         pass
    #                 if len(context):
    #                     try:
    #                         rmi.add_context(context, index=0)
    #                     except:
    #                         pass
    #             #break
    #         rmi.write_model_to_file("/home/tobias/Dokumente/saved_context_vectors/small_word_sim_k5.model")
    #         #break




    # # @depr
    # def txt_file_by_context(rmi, path =""):
    #     """
    #     context ~ file -> same as below but with foldernames as units.
    #     e.g. newsgroups: sports
    #     :param rmi:
    #     :param path:
    #     :param contextSize:
    #     :param ext:
    #     :return:
    #     """
    #     files = get_paths_of_files(path, filetype="")
    #     size = len(files)
    #     for i,filename in zip(range(len(files)),files):
    #         if i%10 == 0:
    #             print("\r%f %%" % (100*i/size), end="")
    #         raw_text = ""
    #         try:
    #             with open(filename, 'r', encoding='utf-8') as fin:
    #                 raw_text = fin.read()
    #         except:
    #             try:
    #                 with open(filename, 'r', encoding='iso-8859-1') as fin:
    #                     raw_text = fin.read()
    #             except:
    #                 print("error with ", filename)
    #                 return
    #         p = Path(filename)
    #         # setting unit!
    #         rmi.add_unit(unit=p.parts[-2], context=(clean_word_seq(w_tokenize(raw_text))))
    #     print("finished news-model")



    # """
    #             BUILD-MODEL-FUNCTIONS
    #             2) Context-Similarity
    # """
    # def file_context(rmi, path = ""):
    #     """
    #      context ~ file -> and assign e.g. each author one context-vec
    #      :param: rmi
    #     :param path:
    #     :return:
    #     """
    #     import re
    #     files = get_paths_of_files(path, filetype=".txt")
    #     size = len(files)
    #     tags = ["<URL>", "<Country>", "<Publication_year>", "<Publisher>", "<Contributor>", "<Author>"]
    #     for i,filename in zip(range(len(files)),files):
    #         if i%10 == 0:
    #             print("\r%f %%" % (100*i/size), end="")
    #         try:
    #             with open(filename, 'r', encoding='utf-8') as fin:
    #                 raw_text = fin.read()
    #         except:
    #             try:
    #                 with open(filename, 'r', encoding='iso-8859-1') as fin:
    #                     raw_text = fin.read()
    #             except:
    #                 print("error with ", filename)
    #                 return
    #         author = re.findall(tags[2] + '[^<]*', raw_text)[0][len(tags[2]):]
    #         article = re.findall('</Author>.*', raw_text,flags=re.DOTALL)[0][len('</Author>'):]
    #         rmi.add_unit(unit=author, context=(clean_word_seq(w_tokenize(article))))
    #     rmi.write_model_to_file("year.model")
    #     print("done.")
    #
    # """
    #             BUILD-MODEL-FUNCTIONS
    #             3) Word-(Letter-) Similarity
    # """
    # def yet_another_func(rmi, path = ""):
    #     with open(path, 'r', encoding='utf-8') as fin:
    #         text = fin.read()
    #     words = w_tokenize(text)
    #     for word in words:
    #         try:
    #             l_word = [letter.lower() for letter in word if letter.isalpha() and len(word)>0]
    #             rmi.add_unit(word.lower(), l_word)
    #         except:
    #             continue