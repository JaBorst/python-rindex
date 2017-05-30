
import sys
sys.path.append('../')

from nltk.tokenize import word_tokenize as w_tokenize
from nltk.tokenize import sent_tokenize as s_tokenize

import re
import os
import pickle
import time
from t_helper import get_paths_of_files
from t_helper import clean_word_seq
from t_helper import normalize_matrix
from t_helper import search_in_matrix

from rindex import *

import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import KDTree

import matplotlib.pyplot as plt


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
    :param method: 
    :param n: 
    :param word: 
    :return: 
    """

    some_element = np.array(kdt.data[keys.index(word)])
    sum_time = 0.0
    for i in range(10):
        start = time.time()
        if method == "query":
            dist, ind = kdt.query(some_element.reshape(1, -1), k=n)
        sum_time += time.time() - start
    print(sum_time/10)
    # elif method == "radius":
    #     ind = kdt.query_ball_point(x=some_element, r=1)
    #print("searching tree for {0} took me {1: >#08.5f} sec.".format(n, time.time() - start))

    for i, d in zip(ind[0], dist[0]):
        print("{0}\t{1: >#016.4f}".format(keys[i], d))


def analyze_file_by_context(filename, rmi, context_size=2):
    """
    
    :param filename: 
    :param rmi: 
    :param context_size: 
    :return: 
    """
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
        # if i%100 == 0:
        #     print("\r%f %%" % (100*i/size), end="")
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
    #print("finish")


def analyze_text_files_of_folder(rmi, path="", contextSize = 2, ext =""):
    """
    walks through folder and builds a model
    :param:rmi
    :param path: 
    :param contextSize: 
    :param ext: 
    :return: 
    """

    files = get_paths_of_files(path,filetype=".txt")
    size = len(files)
    print(size)
    for i, filename in zip(range(len(files)),files):
        if i%10 == 0:
            print("\r%f %%" % (100*i/size), end="")
        analyze_file_by_context(filename, rmi, context_size=contextSize)
        if i%50 == 0:
            rmi.write_model_to_file(ext)
    rmi.write_model_to_file(ext)


def build_parteiprogramm_model(rmi, path=""):
    import PyPDF2
    from os.path import basename

    files = get_paths_of_files(path, filetype=".pdf")
    size = len(files)
    for i,filename in zip(range(len(files)),files):
        print("\r%f %%" % (100*i/size), end="")
        pobj = open(filename, 'rb')
        pdfReader = PyPDF2.PdfFileReader(pobj)
        full_text = ""
        try:
            for page_number in range(pdfReader.numPages):
                p = pdfReader.getPage(page_number)
                full_text += p.extractText()
            print(basename(filename)[:-4])
            rmi.add_unit(unit=basename(filename)[:-4], context=clean_word_seq(w_tokenize(full_text)))
        except:
            print("error",filename)
    rmi.write_model_to_file("/home/tobias/Dokumente/saved_context_vectors/parteiprogramm_30000.model")


def txt_file_by_context(rmi, path ="", contextSize=2,ext=""):
    """
    
    :param rmi: 
    :param path: 
    :param contextSize: 
    :param ext: 
    :return: 
    """
    from pathlib import Path
    files = get_paths_of_files(path, filetype="")
    size = len(files)
    for i,filename in zip(range(len(files)),files):
        if i%10 == 0:
            print("\r%f %%" % (100*i/size), end="")
        raw_text = ""
        try:
            with open(filename, 'r', encoding='utf-8') as fin:
                raw_text = fin.read()
        except:
            try:
                with open(filename, 'r', encoding='iso-8859-1') as fin:
                    raw_text = fin.read()
            except:
                print("error with ", filename)
                return
        p = Path(filename)
        rmi.add_unit(unit=p.parts[-2], context=(clean_word_seq(w_tokenize(raw_text))))
    rmi.write_model_to_file("news.model")
    print("done.")


def file_context(rmi, path = ""):
    """
     context ~ file -> read text-files
     :param: rmi
    :param path: 
    :return: 
    """
    import re
    files = get_paths_of_files(path, filetype=".txt")
    size = len(files)
    tags = ["<URL>", "<Country>", "<Publication_year>", "<Publisher>", "<Contributor>", "<Author>"]
    for i,filename in zip(range(len(files)),files):
        if i%10 == 0:
            print("\r%f %%" % (100*i/size), end="")

        try:
            with open(filename, 'r', encoding='utf-8') as fin:
                raw_text = fin.read()
        except:
            try:
                with open(filename, 'r', encoding='iso-8859-1') as fin:
                    raw_text = fin.read()
            except:
                print("error with ", filename)
                return
        author = re.findall(tags[2] + '[^<]*', raw_text)[0][len(tags[2]):]
        article = re.findall('</Author>.*', raw_text,flags=re.DOTALL)[0][len('</Author>'):]
        rmi.add_unit(unit=author, context=(clean_word_seq(w_tokenize(article))))
    rmi.write_model_to_file("year.model")
    print("done.")


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


def main():
    dim = 1500
    k = 3
    rmi = RIModel.RIModel(dim, k)
    context_size = 3
    rmi.is_sparse = True
    file_source = "/home/tobias/Dokumente/testdata/wortschatz_large/eng_news_2005_1M-sentences.txt"
    folder_source = "/home/tobias/Dokumente/testdata/Newspapers/CLOB_RAW"

    """
        build models
    """
    #analyze_text_files_of_folder(rmi,folder_source,context_size,"/home/tobias/Dokumente/saved_context_vectors/clob_crown/1500_k3_c3.model")
    #txt_file_by_context(rmi, path=folder_source,contextSize=2,ext="/home/tobias/Dokumente/saved_context_vectors/news.model")
    #analyze_file_by_context(filename=file_source, rmi=rmi, context_size=context_size)
    #rmi.write_model_to_file("/home/tobias/Dokumente/saved_context_vectors/wortschatz/dim1500k3c4.model")

    #build_word_sim_model(rmi=rmi, path="/home/tobias/Dokumente/testdata/wortschatz_small",context_size=context_size)
    #build_parteiprogramm_model(rmi=rmi,path=folder_source)

    """
        load models
    """
    #rmi.load_model_from_file('/home/tobias/Dokumente/saved_context_vectors/wortschatz/dim1500k3c4.model')
    #rmi.load_model_from_file('/home/tobias/Dokumente/saved_context_vectors/clob_crown/1500_k3_c3.model')
    #rmi.load_model_from_file('/home/tobias/Dokumente/saved_context_vectors/parteiprogramm.model')
    #print(rmi.ContextVectors)

    """
        Dim-Red
    """
    #rmi.reduce_dimensions(method="truncated_svd", target_size=50)
    #rmi.write_model_to_file('/home/tobias/Dokumente/saved_context_vectors/newsgroups/50accu_no_stemming.model')
    # rmi.write_model_to_file("/home/tobias/Dokumente/saved_context_vectors/oanc/stemming_d50written.model")

    """
        tsne
    """
    #rmi.reduce_dimensions(method="tsne", target_size=2)
    #rmi.is_similar_to(word="man")
    # keys, matrix = rmi.to_matrix()
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # plt.plot(matrix[:,0],matrix[:,1],'ro')
    #
    # for i,x,y in zip(range(len(keys)),matrix[:,0],matrix[:,1]):  # <--
    #     ax.annotate('%s' % keys[i],xy=(x,y),textcoords='data')  # <--
    #
    # plt.grid()
    # plt.show()


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




    """
        Kd-Baum
    """

    # keys, kdt = rmi.to_tree(method="minkowski", leaf_size=50)
    #
    # with open("/home/tobias/Dokumente/saved_context_vectors/word_sim_4c.tree", 'wb') as output:
    #     pickle.dump(kdt, output)
    # with open("/home/tobias/Dokumente/saved_context_vectors/word_sim_4c.keys", 'wb') as output:
    #     pickle.dump(keys, output)

    with open("/home/tobias/Dokumente/saved_context_vectors/word_sim_4c.keys", 'rb') as inputFile:
        keys = pickle.load(inputFile)
    with open("/home/tobias/Dokumente/saved_context_vectors/word_sim_4c.tree", 'rb') as inputFile:
        kdt = pickle.load(inputFile)

    # from nltk.stem.snowball import SnowballStemmer
    # word = SnowballStemmer("english",ignore_stopwords=False).stem("man")
    # search_tree_for_similar(kdt, keys, method="query", n=20, word=word)

    # with open("/home/tobias/Dokumente/saved_context_vectors/paratest/word_sim.keys", 'rb') as inputFile:
    #     keys = pickle.load(inputFile)
    # with open("/home/tobias/Dokumente/saved_context_vectors/paratest/word_sim.tree", 'rb') as inputFile:
    #     kdt = pickle.load(inputFile)
    #
    #for i in range(10):

    search_tree_for_similar(kdt, keys, method="query", n=10, word="news")


if __name__ == '__main__':
    main()
