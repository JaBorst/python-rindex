
from nltk.tokenize import word_tokenize as w_tokenize
from nltk.tokenize import sent_tokenize as s_tokenize
import re
import os
import pickle
import time
from RIModel import RIModel
import numpy as np


def clean_word_seq(context = []):
    """
    # some trimming
    :param context: 
    :return: 
    """
    if len(context) > 1:
        return([word.lower() for word in context if re.search(r'[a-zA-Z]', word) is not None])
    elif len(context) == 0:
        return
    elif len(context) == 1:
        if re.search(r'[a-zA-Z]', context[0]) is not None:
            return context[0].lower()
        else:
            return


        
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
    i = 0
    doc_iv = rmi.iv.create_index_vector_from_context(filename)
    for sentence in content:
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
                        ## add something for each doc extrax
                        #rmi.ContextVectors[context[0]] += doc_iv
                    except:
                        pass
        except:
            pass            
        i += 1


def get_paths_of_files(path):
    """
    
    :param path: 
    :return: 
    """
    list_of_files = []
    i = 0
    for (dirpath, dirnames, filenames) in os.walk(path):
        for filename in filenames:
            if filename.endswith('.txt'): 
                list_of_files.append(os.sep.join([dirpath, filename]))
    return list_of_files


# @todo: anderer name
def analyze_files_of_folder(path="", contextSize = 2, ext = ""):
    """
    walks through folder and builds a model
    ideas:
    - add a random (random) vector for each document
      to each context in this document
    - build the models incremental, i.e. one model for
      each file, then add them up -> is this possible?
    -> at the moment it's just one model, also ok
    :param path: 
    :param contextSize: 
    :param ext: 
    :return: 
    """

    files = get_paths_of_files(path)
    rmi = RIModel(dim = 1500, k = 3)
    i = 0
    size = len(files)
    print(size)
    for filename in files:
        if i%10 == 0:
            print("\r%f %%" % (100*i/size), end="")

        analyze_file_by_context(filename, rmi, contextSize = 2)
        if i%50 == 0:
            ## maybe just for testing save the model once in a
            ## while: could save some nerves.
            rmi.write_model_to_file(ext)
        i += 1
    rmi.write_model_to_file(ext)


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

def to_tree(rim):
    """
    should be done !only! with reduced data
    to enhence search
    :param rim: 
    :return: 
    """
    from sklearn.neighbors import KDTree
    key= "france" #test

    values = list(rim.ContextVectors.values())
    keys = list(rim.ContextVectors.keys())
    leafs = np.zeros((len(values),rim.dim))

    if not rim.is_sparse:
        leafs = [val for val in values]
        some_element = rim.ContextVectors[key]
    else:
        leafs = [val.toarray for val in values]
        some_element = rim.ContextVectors[key]

    start = time.time()
    kdt = KDTree(leafs, leaf_size=50, metric='minkowski')
    # search returns index of source-array, i.e. vals
    print("building tree took me {0: >#08.4f} sec.".format(time.time()-start))

    n_neighbours = 10
    start = time.time()
    # converts (n,) to (1,n)
    some_element = some_element.reshape(1,-1)
    dist, ind = kdt.query(some_element, k=n_neighbours)

    print("searching tree for {0} took me {1: >#08.5f} sec.".format(n_neighbours,time.time()-start))
    print("\nword | dist")
    for i,d in zip(ind[0],dist[0]):
        print("{0}\t{1: >#016.4f}".format(keys[i],d))

    filename = "/home/tobias/tree.pkl"
    # with open(filename, 'wb') as output:
    #     pickle.dump(kdt, output)

    # filename = "/home/tobias/keys.pkl"
    # with open(filename, 'wb') as output:
    #     pickle.dump(keys, output)


def main():
    dim = 50
    k = 3
    rmi = RIModel(dim, k)
    context_size = 2
    rmi.is_sparse = True
    file_source = "/home/tobias/Dokumente/testdata/stateofunion.txt"
    folder_source = "/home/tobias/Downloads/OANC/data/written_1"

    #analyze_files_of_folder(path=folder_source,contextSize=2,ext="written_1")
    #analyze_file_by_context(filename=file_source,rmi=rmi, contextSize=context_size)
    #rmi.write_model_to_file("sou_5")
    rmi.load_model_from_file('/home/tobias/Dokumente/saved_context_vectors/oanc/d50k3svd_written_1.pkl')
    #rmi.reduce_dimensions(method="truncated_svd", target_size=75)
    #rmi.write_model_to_file("svd_written_1")

    """
        Vektor-Arithmetik
    """
    # print(rmi.vector_add(words=["man","kid"],isword="boy")) #0.0623583290773
    # print(rmi.vector_add(words=["waffle", "kid"], isword="church")) #0.0.0535538218819
    # mit euclicdean
    # 1268.009495090077
    # 181.04640127229905

    to_tree(rmi)
    #rmi.is_similar_to(word="man", thres=0.9, count=10)







if __name__ == '__main__':
    main()



# deprecated
# def analyze_file_by_sentence(filename, rmi):
#     """
#     Simple model which first reads the texts as sentences
#     and then takes the first word as context.
#     """
#     with open(filename) as fin:
#         text = fin.read()
#     content = s_tokenize(text)
#
#     size = len(content)
#     i = 0
#     for sentence in content:
#         if i%100 == 0:
#             print("\r%f %%" % (100*i/size), end="")
#         # take first word of sentence -> index = 0
#         rmi.add_context(w_tokenize(sentence), index = 0)
#         i += 1
