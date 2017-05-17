
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
from rindex import *

import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import KDTree

"""
    @todo Move to RIModel. Ich finde, zumindest die tree Methode können wir 
    auch außerhalb lassen. Das hat ja so gesehen nichts mehr mit dem Model zu tun.
    begin
"""
# def to_matrix(rim, to_sparse=False, is_sparse=True):
#     """
#
#     :param rim:
#     :param sparse:
#     :return:
#     """
#     keys = rim.ContextVectors.keys()
#     i = range(len(keys))
#     if to_sparse and is_sparse:
#         target_martix = sp.lil_matrix((len(keys), rim.dim))
#         for key, i in zip(keys,i):
#             target_martix[i] = rim.ContextVectors[key].transpose()
#     elif not to_sparse:
#         target_martix = np.zeros((len(keys), rim.dim))
#         if is_sparse:
#             for key, i in zip(keys,i):
#                 target_martix[i, :] = rim.ContextVectors[key].transpose().toarray()
#         else:
#             for key, i in zip(keys,i):
#                 target_martix[i, :] = rim.ContextVectors[key]
#     print("converted dict to matrix")
#     return list(keys), target_martix


def normalize_matrix(matrix):
    from sklearn.preprocessing import normalize
    return normalize(matrix, axis=1, norm='l1')


def search_in_matrix(matrix,keys=[],word=""):
    """
    
    :param matrix: 
    :param keys: 
    :param word: 
    :return: 
    """
    from scipy import spatial
    word_iv = matrix[keys.index(word)].toarray()
    max_d = 0
    max_key = ""
    for key in keys:
        if key != word:
            d = 1-spatial.distance.cosine(word_iv, matrix[keys.index(key)].toarray())
            if d > max_d:
                max_d = d
                max_key = key
    print(max_d, max_key)


# def to_tree(rim, method='minkowski'):
#     """
#     should be done !only! with reduced data
#     to enhence search
#     :param rim:
#     :return:
#     """
#     if rim.dim > 50:
#         print(">50 dim is not recommended. please reduce first.")
#         return
#     if not rim.is_sparse:
#         keys, leafs = to_matrix(rim, to_sparse=False, is_sparse=False)
#     else:
#         keys, leafs = to_matrix(rim, to_sparse=False, is_sparse=True)
#     leafs = normalize_matrix(leafs)#####
#     return keys, KDTree(leafs, leaf_size=50, metric=method)


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

"""
    end
    @todo Move to RIModel
"""
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
    content = s_tokenize(text)
    size = len(content)

    for i, sentence in zip(range(len(context)),content):
        if i%100 == 0:
            print("\r%f %%" % (100*i/size), end="")
        ## man kann natürlich auch hier das 1. wort rausnehmen
        ## für jedes wort
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

    files = get_paths_of_files(path)
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
    dim = 3000
    k = 5
    rmi = RIModel.RIModel(dim, k)
    context_size = 2
    rmi.is_sparse = True
    file_source = "/home/tobias/Dokumente/testdata/stateofunion.txt"
    folder_source = "/home/tobias/Dokumente/testdata/wp_entwürfe_2017"

    # analyze_files_of_folder(path=folder_source,contextSize=2,ext="written_1")
    # analyze_file_by_context(filename=file_source,rmi=rmi, contextSize=context_size)
    # rmi.write_model_to_file("sou_5")
    #rmi.load_model_from_file('/home/tobias/Dokumente/saved_context_vectors/paratest/accu.model')
    # rmi.write_model_to_file("svd_written_1")

    #build_word_sim_model(rmi=rmi, path="/home/tobias/Dokumente/testdata/wortschatz_small",context_size=context_size)
    #build_parteiprogramm_model(rmi=rmi,path=folder_source)

    rmi.load_model_from_file('/home/tobias/Dokumente/saved_context_vectors/paratest/accu.model')
    #print(rmi.ContextVectors)

    #rmi.is_similar_to(word="man", thres=0.6, count=10)
    #file_context(rmi=rmi, path="/home/tobias/Dokumente/testdata/Newspapers/Crown_with_metadata")
    # keys, matrix = to_matrix(rim=rmi)
    # normed_matrix= normalize_matrix(matrix)
    # search_in_matrix(matrix=matrix,keys=keys, word="man")
    #rmi.is_similar_to(word="e2017_afd",thres = 0.1, count = 10)

    """
        geht leider nicht
    """
    # tag_names_to_exclude = {}
    # reader = Project1Filter(tag_names_to_exclude, make_parser())
    # filename = "/home/tobias/Dokumente/testdata/Newspapers/CLOB_with_metadata/A01BA.txt"
    # with open('out-small.xml', 'w') as f:
    #     handler = XMLGenerator(f)
    #     reader.setContentHandler(handler)
    #     reader.parse(filename)

    """
        Dim-Red
    """
    rmi.reduce_dimensions(method="truncated_svd", target_size=20)
    #rmi.reduce_dimensions(method="", target_size=2)

    #rmi.is_similar_to(word="man",thres=0.9, count=10)

    """
        Vektor-Arithmetik
    """
    # print(rmi.vector_add(words=["man","kid"],isword="boy")) #0.0623583290773
    # print(rmi.vector_add(words=["waffle", "kid"], isword="church")) #0.0.0535538218819

    """
        Kd-Baum (als Funktion)
    """
    #
    # keys, kdt = rmi.to_tree(method="minkowski", leaf_size=50)
    # with open("/home/tobias/Dokumente/saved_context_vectors/word_sim.tree", 'wb') as output:
    #     pickle.dump(kdt, output)
    # with open("/home/tobias/Dokumente/saved_context_vectors/word_sim.keys", 'wb') as output:
    #     pickle.dump(keys, output)

    # with open("/home/tobias/Dokumente/saved_context_vectors/word_sim.keys", 'rb') as inputFile:
    #     keys = pickle.load(inputFile)
    # with open("/home/tobias/Dokumente/saved_context_vectors/word_sim.tree", 'rb') as inputFile:
    #     kdt = pickle.load(inputFile)
    #
    # search_tree_for_similar(kdt, keys, method="query", n=10, word="man")


if __name__ == '__main__':
    main()
