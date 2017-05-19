
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
    start = time.time()
    if method == "query":
        dist, ind = kdt.query(some_element.reshape(1, -1), k=n)
    # elif method == "radius":
    #     ind = kdt.query_ball_point(x=some_element, r=1)
    print("searching tree for {0} took me {1: >#08.5f} sec.".format(n, time.time() - start))

    for i, d in zip(ind[0], dist[0]):
        print("{0}\t{1: >#016.4f}".format(keys[i], d))


def analyze_file_by_context(filename, rmi, contextSize = 2):
    """
    
    :param filename: 
    :param rmi: 
    :param contextSize: 
    :return: 
    """
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
    # first extract sentences (later tokenize to words)
    context = s_tokenize(text)
    size = len(context)
    for i, sentence in zip(range(len(context)),context):
        # if i%100 == 0:
        #     print("\r%f %%" % (100*i/size), end="")

        sent = clean_word_seq(w_tokenize(sentence))
        try:
            # kann auch bis len(sent)-contextSize gehen
            for j in range(len(sent)):
                context = []
                ## schnappe dir das nächste(n) wort(e), wenn es das gibt
                for k in range(j, j+contextSize):
                    try:
                        context.append(sent[k])
                    except:
                        pass
                ## und füge den Kontext hinzu
                if len(context):
                    try:
                        rmi.add_context(context, index = 0)
                    except:
                        pass
        except:
            pass            
    #print("finish")


def analyze_text_files_of_folder(rmi, path="", contextSize = 2, ext =""):
    """
    walks through folder and builds a model
    ideas:
    - add a random (random) vector for each document
      to each context in this document
    - build the models incremental, i.e. one model for
      each file, then add them up -> is this possible?
    -> at the moment it's just one model, also ok
    :param:rmi
    :param path: 
    :param contextSize: 
    :param ext: 
    :return: 
    """

    files = get_paths_of_files(path,filetype="")
    size = len(files)
    print(size)
    for i, filename in zip(range(len(files)),files):
        if i%10 == 0:
            print("\r%f %%" % (100*i/size), end="")
        analyze_file_by_context(filename, rmi, contextSize = 2)
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


def build_word_sim_model(rmi, path="", context_size=2):
    """
    walks through all of the files 2006-2015 (eng) to build a 
    model for word simi.
    :param rmi: 
    :param path: 
    :param context_size: 
    :return: 
    """
    files = get_paths_of_files(path, filetype=".txt")
    #files = ["/home/tobias/Dokumente/testdata/eng_news_2015_10K-sentences.txt"]
    for file in files:
        with open(file, 'r', encoding="utf-8") as fin:
            sents = fin.readlines()
        print(file)
        size = len(sents)
        for i,sent in zip(range(len(sents)),sents):
            if i % 100 == 0:
                print("\r%f %%" % (100 * i / size), end="")
            sent = clean_word_seq(w_tokenize(sent))
            for j in range(len(sent)):
                context = []
                for k in range(j, j + context_size):
                    try:
                        context.append(sent[k])
                    except:
                        pass
                if len(context):
                    try:
                        rmi.add_context(context, index=0)
                    except:
                        pass
            #break
        rmi.write_model_to_file("/home/tobias/Dokumente/saved_context_vectors/small_word_sim_k5.model")
        #break


def main():
    dim = 1500
    k = 3
    rmi = RIModel.RIModel(dim, k)
    context_size = 2
    rmi.is_sparse = True
    file_source = "/home/tobias/Dokumente/testdata/20_newsgroups/comp.graphics/"
    folder_source = "/home/tobias/Dokumente/testdata/20_newsgroups/"

    #txt_file_by_context(rmi, path=folder_source,contextSize=2,ext="/home/tobias/Dokumente/saved_context_vectors/news.model")
    #analyze_file_by_context(filename=file_source,rmi=rmi, contextSize=context_size)
    # rmi.write_model_to_file("svd_written_1")

    #build_word_sim_model(rmi=rmi, path="/home/tobias/Dokumente/testdata/wortschatz_small",context_size=context_size)
    #build_parteiprogramm_model(rmi=rmi,path=folder_source)

    rmi.load_model_from_file('/home/tobias/Dokumente/saved_context_vectors/paratest/merge.model')
    print(rmi.ContextVectors)

    #rmi.is_similar_to(word="man", thres=0.9, count=10)
    #file_context(rmi=rmi, path="/home/tobias/Dokumente/testdata/Newspapers/Crown_with_metadata")
    #normed_matrix = normalize_matrix(matrix)

    #rmi.is_similar_to(word="man", thres=0.1, count=10)

    #print(rmi.ContextVectors.keys())

    """
        Dim-Red
    """
    rmi.reduce_dimensions(method="truncated_svd", target_size=50)
    #rmi.reduce_dimensions(method="mds", target_size=2)
    # #rmi.is_similar_to(word="2012_partei_der_vernunft", thres=0.1, count=10)
    #keys, matrix = rmi.to_matrix()

    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # plt.plot(matrix[:,0],matrix[:,1],'ro')
    #
    # for i,x,y in zip(range(len(keys)),matrix[:,0],matrix[:,1]):  # <--
    #     ax.annotate('%s' % keys[i],xy=(x,y),textcoords='data')  # <--
    #
    # plt.grid()
    # plt.show()
    #

    """
        Vektor-Arithmetik
    """
    # print(rmi.vector_add(words=["man","kid"],isword="boy")) #0.0623583290773
    # print(rmi.vector_add(words=["waffle", "kid"], isword="church")) #0.0.0535538218819

    """
        Kd-Baum (als Funktion)
    """

    keys, kdt = rmi.to_tree(method="minkowski", leaf_size=50)
    #
    # with open("/home/tobias/Dokumente/saved_context_vectors/word_sim2.tree", 'wb') as output:
    #     pickle.dump(kdt, output)
    # with open("/home/tobias/Dokumente/saved_context_vectors/word_sim2.keys", 'wb') as output:
    #     pickle.dump(keys, output)

    # with open("/home/tobias/Dokumente/saved_context_vectors/word_sim.keys", 'rb') as inputFile:
    #     keys = pickle.load(inputFile)
    # with open("/home/tobias/Dokumente/saved_context_vectors/word_sim.tree", 'rb') as inputFile:
    #     kdt = pickle.load(inputFile)

    search_tree_for_similar(kdt, keys, method="query", n=10, word="man")

    # with open("/home/tobias/Dokumente/saved_context_vectors/word_sim.keys", 'rb') as inputFile:
    #     keys = pickle.load(inputFile)
    # with open("/home/tobias/Dokumente/saved_context_vectors/word_sim.tree", 'rb') as inputFile:
    #     kdt = pickle.load(inputFile)
    #
    # search_tree_for_similar(kdt, keys, method="query", n=10, word="women")


if __name__ == '__main__':
    main()
