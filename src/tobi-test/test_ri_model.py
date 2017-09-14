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

# import re
# import os
import pickle
import numpy as np
# import scipy.sparse as sp
# from sklearn.neighbors import KDTree

import matplotlib.pyplot as plt

# from pathlib import Path
# from os.path import basename
# from t_helper import get_paths_of_files

from rindex import *
from t_helper import write_dicct_to_csv
# used for nice printings
line_length= 60
dotted_line = "-"*line_length
line = "_"*line_length
colon_line = ":"*line_length
quarter_space_line = " "*int(line_length/4)


def visualize_vecs(rmi, vec_names=[]):
    """
    get a graphic representation of cvs (for testing)
    :param rmi:
    :param vec_names:
    :return:
    """
    if len(vec_names) == 0:
        print("no vecs specified. good luck.")
        vec_names = [key for key in rmi.ContextVectors.keys()]
        print(vec_names)
    ivs = []
    if rmi.is_sparse:
        for name in vec_names:
            ivs.append(rmi.ContextVectors[name].toarray())
    else:
        for name in vec_names:
            ivs.append(rmi.ContextVectors[name])
    x = [i for i in range(ivs[0].shape[0])]
    f, axarr = plt.subplots(len(vec_names), sharex=True)
    for i in range(len(vec_names)):
        axarr[i].plot(x, ivs[i])
        #axarr[i].set
        axarr[i].set_title(vec_names[i])
        axarr[i].set_ylabel('value')
    axarr[-1].set_xlabel('vec position')
    plt.show()


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



#
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




def main():
    dim = 1500
    k = 3
    rmi = RIModel.RIModel(dim, k)
    #rmi.is_sparse = True

    print(colon_line)
    print(quarter_space_line,"Random Indexing Model")
    print(colon_line)
    mode = input("""\t[0]: compare \t 2: visualize\n\t 
    3: clustering\t 4: search tree\n""") or 0
    reduction = input("\ty do reduce \t [n] don't reduce") or 'n'

    """
        load models
    """
    model_filename = "/home/tobias/Dokumente/models/doc-models/plato-test.model"
        #"/home/tobias/Dokumente/models/doc-models/books/no_editorial_stem_both_d1500k3_c4i0.model"
    rmi.load_model_from_file(model_filename)

    print(rmi.dim)
    #print(rmi.ContextVectors)

    if reduction == 'y':
        rmi.reduce_dimensions(method="truncated_svd", target_size=len(rmi.ContextVectors))
        # rmi.reduce_dimensions(method="tsne", target_size=2)

    # je niedriger, desto mehr Werte bleiben erhalten
    # rmi.truncate(threshold=0.0)
    # rmi.is_sparse = False



    #print(rmi.ContextVectors)
    if int(mode) == 0:
        rmi.is_similar_to(word="plato_apology", count=15, method="cos",silent=False)
        #is_equal_heap = rmi.most_similar(count=-1,method="jsd",silent=True)
        #write_dicct_to_csv(is_equal_heap,
        #                   output_file="/home/tobias/Dokumente/models/doc-models/books/logs/log.csv")

    elif int(mode) == 2:
        vec_names = ['shakespeare_hamlet', 'shakespeare_henry_the_fifth',
                     #'plato_symposium','goethe_faust', 'tolstoy_war_and_peace', 'tolstoy_anna_karenina',
                     'melville_moby_dick','austen_emma']
        visualize_vecs(rmi, vec_names=[])
    elif int(mode) == 3:
        """
            Themen Clustering
        """
        # mit dim-red und normalize!
        from sklearn.cluster import KMeans
        import collections
        keys, matrix = rmi.to_matrix(to_sparse=False)
        km_model = KMeans(n_clusters=7)
        km_model.fit(matrix)
        clustering = collections.defaultdict(list)
        for idx, label in enumerate(km_model.labels_):
            clustering[label].append(idx)
        for key, value in clustering.items():
            print(key, [keys[idx] for idx in value])

    elif int(mode) == 4:
        """
            Kd-Tree
        """
        tree_filename = "/home/tobias/Dokumente/models/trees/word_sim_3c.tree"
        leafs_filename = "/home/tobias/Dokumente/models/trees/word_sim_3c.keys"
        build_tree = input("\ty build tree \t [n] don't build tree") or 'n'

        if build_tree == 'y':
            keys, kdt = rmi.to_tree(method="minkowski", leaf_size=50)
            with open(tree_filename, 'wb') as output:
                pickle.dump(kdt, output)
            with open(leafs_filename, 'wb') as output:
                pickle.dump(keys, output)
        elif build_tree == 'n':
            with open(tree_filename, 'rb') as inputFile:
                kdt = pickle.load(inputFile)
            with open(leafs_filename, 'rb') as inputFile:
                keys = pickle.load(inputFile)
        else:
            return

        while True:
            search_word = input("search for word:\n")

            try:
                # from nltk.stem.snowball import SnowballStemmer
                # search_word = SnowballStemmer("english",ignore_stopwords=False).stem(search_word)
                search_tree_for_similar(kdt, keys, method="query", n=7, word=search_word)
            except:
                print("word not in contextvectors.")
            if search_word == "stop":
                break


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
