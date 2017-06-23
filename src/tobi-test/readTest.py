
import sys
sys.path.append('../')

from nltk.tokenize import word_tokenize as w_tokenize
from nltk.tokenize import sent_tokenize as s_tokenize

import re
import os
import pickle
import time

import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import KDTree

import matplotlib.pyplot as plt
from pathlib import Path

## for pdfs
import PyPDF2
from os.path import basename


from t_helper import get_paths_of_files
from t_helper import clean_word_seq
from t_helper import get_n_sents
#from t_helper import normalize_matrix
#from t_helper import search_in_matrix

from rindex import *


# used for nice printings
line_length= 60
dotted_line = "-"*line_length
line = "_"*line_length
colon_line = ":"*line_length
quarter_space_line = " "*int(line_length/4)


def get_weight_vector(context_size=2, target_index=1):
    """
    e.g. 0.75 0 0.75 0.25
    :param context_size: 
    :param target_index: 
    :return: 
    """
    weights=np.zeros(context_size)
    step_size = 1 / context_size
    weights[:target_index] = np.linspace(1 - step_size, 1, num=target_index)
    weights[target_index + 1:] = np.linspace(1, step_size, num=context_size-target_index - 1)
    return  weights

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
    #sum_time = 0.0
    #for i in range(1):
    #start = time.time()
    if method == "query":
        dist, ind = kdt.query(some_element.reshape(1, -1), k=n)
    #sum_time += time.time() - start
    #print(sum_time/10)
    for i, d in zip(ind[0], dist[0]):
        print("{0}\t{1: >#016.4f}".format(keys[i], d))


"""
            BUILD-MODEL-FUNCTIONS
            1) Word-Similarity
"""
def analyze_file_by_context(filename, rmi, context_size=2):
    """
    
    :param filename: 
    :param rmi: 
    :param context_size: 
    :return: 
    """
    #print("calls add_context!")
    index = 1
    weights = get_weight_vector(context_size, target_index=index)
    try:
        with open(filename,'r' ,encoding='utf-8') as fin:
            text = fin.read()
    except:
        try:
            with open(filename,'r' ,encoding='iso-8859-1') as fin:
                text = fin.read()
        except:
            print("error with ",filename)
            return
    context = s_tokenize(text)
    size = len(context)
    for i, sentence in zip(range(len(context)),context):
        if i%100 == 0:
            print("\r%f %%" % (100*i/size), end="")
        sent = clean_word_seq(w_tokenize(sentence))
        try:
            for j in range(len(sent)):
                context = []
                for k in range(j, j+context_size):
                    try:
                        context.append(sent[k])
                    except:
                        pass
                if len(context):
                    try:
                        # first creates iv by contex, second incr.
                        #rmi.add_context(context, index=index)
                        rmi.add_unit(context[index], context, weights=weights)
                    except:
                        pass
        except:
            pass
    #print("finished analyze_file_by_context")


def analyze_text_files_of_folder(rmi, source_path="", contextSize = 2, ext =""):
    """
    walks through folder and builds a model
    :param:rmi
    :param path: 
    :param contextSize: 
    :param ext: 
    :return: 
    """
    files = get_paths_of_files(source_path,filetype=".txt")
    if len(files) == 0:
        print("not txt-files in folder")
        return
    size = len(files)
    for i, filename in zip(range(len(files)),files):
        if i%10 == 0:
            print("\r%f %%" % (100*i/size), end="")
        analyze_file_by_context(filename, rmi, context_size=contextSize)
        if i%50 == 0:
            rmi.write_model_to_file(ext)
    rmi.write_model_to_file(ext)
    print("finished analyze_text_files_of_folder-model")

"""
            BUILD-MODEL-FUNCTIONS
            2) Document-Similarity
"""
def build_doc_vec(filename="", dim=1500, k=3, context_size=3, index=1):
    """

    :param filename:
    :param dim:
    :param k:
    :param context_size:
    :param index:
    :return:
    """
    weights = get_weight_vector(context_size, target_index=index)
    print("with weight-vector:",weights)

    with open(filename, 'r', encoding='utf-8') as fin:
        full_text = fin.read()
    try:
        base_filename = basename(filename)#[:-4] ohne file-endung
        print(base_filename)
        rmi = RIModel.RIModel(dim, k)

        doc_text = s_tokenize(full_text)
        #doc_text = get_n_sents(100)
        num_sents = len(doc_text)

        for z, sentence in zip(range(num_sents), doc_text):
            if z % 100 == 0:
                print("\r%f %%" % (100 * z / num_sents), end="")
            sent = clean_word_seq(w_tokenize(sentence))
            try:
                for j in range(len(sent)):
                    context = []
                    for w in range(j, j + context_size):
                        try:
                            context.append(sent[w])
                        except:
                            pass
                    if len(context):
                        try:
                            rmi.add_context(context, index=index)  # for actual context
                            rmi.add_unit(context[index], context, weights=weights)  # for word context
                        except:
                            pass
            except Exception as e:
                pass
        # normalize and cut
        compr = sp.coo_matrix((rmi.dim, 1))
        for key in rmi.ContextVectors.keys():
            compr += rmi.ContextVectors[key]

        return compr
    except Exception as e:
        print("error", filename, e)

def build_doc_model(document_rmi, source_path=""\
                               ,context_size=3, dim=1500, k=10, index=1):

    files = get_paths_of_files(source_path, filetype="")
    size = len(files)
    for i,filename in zip(range(size),files):
        document_rmi.ContextVectors[basename(filename)] = build_doc_vec(filename, dim=dim, k=k,
                                                                        context_size=context_size, index=index)



#vec_to_model(path="/home/tobias/Dokumente/models/doc-models/books", target_file="/home/tobias/Dokumente/models/doc-models/books/books.model")


def vec_to_model(path="", target_file="", dim=1500):
        """

        :param path:
        :param target_file:
        :return:
        """
        model_files = get_paths_of_files(path, filetype=".vec")
        rim1 = RIModel.RIModel(dim=dim, k=3)  # dummy

        print("start merging.")

        size = len(model_files)
        for i in range(1, len(model_files)):
            if i % 10 == 0:
                print("\r%f %%" % (100 * i / size), end="")
            if model_files[i] == target_file:
                continue
            with open(model_files[i] , 'rb') as inputFile:
                rim1.ContextVectors[basename(model_files[i])[:-4]] = pickle.load(inputFile)

        rim1.write_model_to_file(target_file)


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
    mode = input("\t[0]: search \t 1: build\n\t 2: load, reduce and build tree\n") or 0

    if int(mode) == 1:
        """-*
            build models
        """
        file_source = "/home/tobias/Dokumente/testdata/stateofunion.txt"
        folder_source = "/home/tobias/Dokumente/testdata/Newspapers/Crown_RAW"


        document_rmi = RIModel.RIModel(1500, 3)
        build_doc_model(source_path="/home/tobias/Dokumente/testdata/doc_modell" \
                                   ,context_size=2, k=document_rmi.k, dim=document_rmi.dim, index=0\
                        , document_rmi=document_rmi)
        """
            Update Model
        """
        # document_rmi.load_model_from_file("/home/tobias/Dokumente/models/doc-models/books/testing_books.model")
        # filename = "/home/tobias/Dokumente/testdata/doc_modell/shakespeare_romeo_and_juliet"
        # document_rmi.ContextVectors[basename(filename)] = build_doc_vec(filename, dim=document_rmi.dim, k=document_rmi.k,
        #                                                                      context_size=3, index=0)

        #analyze_text_files_of_folder(rmi,path=folder_source,ext="/home/tobias/Dokumente/models/written2_1500k3c3.model",contextSize=context_size)
        #analyze_file_by_context(filename=file_source, rmi=rmi, context_size=context_size)
        document_rmi.write_model_to_file("/home/tobias/Dokumente/models/doc-models/books/testing_books.model")


    if int(mode) == 2:
        """
            load models
        """
        filename= "/home/tobias/Dokumente/models/doc-models/books/testing_books.model"
        rmi.load_model_from_file(filename)
        keys, matrix = rmi.to_matrix(to_sparse=True)
        from sklearn.preprocessing import normalize
        values = normalize(matrix, axis=1, norm='l2', copy=False)
        for key, i in zip(keys,range(len(keys))):
            rmi.ContextVectors[key]=values[i,:].transpose()
        print(rmi.ContextVectors)


        rmi.is_similar_to(word="shakespeare_julius_caesar",  thres=0.9, count=10, method="jaccard")

        # """
        #     Dimensionsreduktion
        # """
        #rmi.reduce_dimensions(method="truncated_svd", target_size=6)
        #rmi.is_similar_to(word="goethe_the_sorrows_of_young_werther",  thres=0.9, count=10, method="jsd")

        #rmi.write_model_to_file('/home/tobias/Dokumente/models/complete_merge50k3c3.model')

        """
            Themen Clustering
        """
        from sklearn.cluster import KMeans
        km_model = KMeans(n_clusters=5)
        km_model.fit(matrix)
        import collections
        clustering = collections.defaultdict(list)
        for idx, label in enumerate(km_model.labels_):
            clustering[label].append(idx)
        for key, value in clustering.items():
            print(key, [keys[idx] for idx in value])


    # if int(mode) == 0 or int(mode)== 2:
    #     """
    #         Kd-Tree
    #     """
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