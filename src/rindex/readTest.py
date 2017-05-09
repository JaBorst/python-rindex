
from nltk.tokenize import word_tokenize as w_tokenize
from nltk.tokenize import sent_tokenize as s_tokenize
import re
import time
import os
#import multiprocessing
import pickle

#import IndexVectorSciPy as IndexVec
from RIModel import RIModel
from sklearn.neighbors import KDTree
import numpy as np

# multidimensional scaling
from sklearn import manifold
import scipy.sparse as sp
from matplotlib import pyplot as plt

def clean_word_seq(context = []):
    # some trimming
    if len(context) > 1:
        return([word.lower() for word in context if re.search(r'[a-zA-Z]', word) is not None])
    elif len(context) == 0:
        return
    elif len(context) == 1:
        if re.search(r'[a-zA-Z]', context[0]) is not None:
            return context[0].lower()
        else:
            return


def analyze_file_by_sentence(filename, rmi):
    """
    Simple model which first reads the texts as sentences
    and then takes the first word as context.
    """
    with open(filename) as fin:
        text = fin.read()
    content = s_tokenize(text)

    size = len(content)
    i = 0
    for sentence in content:
        if i%100 == 0:
            print("\r%f %%" % (100*i/size), end="")        
        # take first word of sentence -> index = 0
        rmi.add_context(w_tokenize(sentence), index = 0)
        i += 1


        
def analyze_file_by_context(filename, rmi, contextSize = 2):
    # file-decoding festlegen
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
    list_of_files = []
    i = 0
    for (dirpath, dirnames, filenames) in os.walk(path):
        for filename in filenames:
            if filename.endswith('.txt'): 
                list_of_files.append(os.sep.join([dirpath, filename]))
    return list_of_files


## @todo: anderer name
def analyze_files_of_folder(path, contextSize = 2, ext = ""):
    # walks through folder and builds a model
    # ideas:
    # - add a random (random) vector for each document
    #   to each context in this document
    # - build the models incremental, i.e. one model for
    #   each file, then add them up -> is this possible?
    # -> at the moment it's just one model

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
    # should be done !only! with reduced data
    points = rim.ContextVectors.values()
    i = 0
    vals = []
    for val in points:
        vals.append(val.toarray()[0])
        i += 1
    kdt = KDTree(vals, leaf_size=40, metric='minkowski')
    filename = "/home/tobias/tree.pkl"
    with open(filename, 'wb') as output:
        pickle.dump(kdt, output)

    # lässt mich keine keys abspeichern...
    # filename = "/home/tobias/keys.pkl"
    # with open(filename, 'wb') as output:
    #     pickle.dump(rim.ContextVectors, output)
    #save keys seperately

def main():
    path= "/home/tobias/Dokumente/pyworkspace/rindex/testdata/Newspapers/Crown_RAW"
    #analyzeFilesOfFolder(path, contextSize=2)
    dim = 1500
    k = 3
    #analyze_files_of_folder("/home/tobias/Downloads/OANC/data/written_1",contextSize=2,ext="written_1")
    rmi = RIModel(dim ,k)
    #analyze_file_by_context("/home/tobias/Dokumente/testdata/stateofunion.txt",rmi=rmi,
    #                        contextSize=5)
    #rmi.write_model_to_file("sou_5")
    # ist der witz nicht eigentlich, das die abstände +-gleich bleiben sollen
    rmi.load_model_from_file('/home/tobias/Dokumente/saved_context_vectors/clob_crown/d1500k3merge.pkl')
    #rmi.is_similar_to(word="man", thres=0.9, count=10)

    keys = rmi.ContextVectors.keys()
    numels = len(keys)

    large_matrix = sp.lil_matrix((numels, rmi.dim))
    i = 0
    for key in keys:
        #print(len(rmi.ContextVectors[key].toarray()))
        large_matrix[i] = rmi.ContextVectors[key].transpose()
        i += 1
        if i == numels:
            break
    from sklearn.decomposition import TruncatedSVD
    svd = TruncatedSVD(n_components=50, n_iter=10, random_state=42)
    svd.fit(large_matrix)
    print(svd.explained_variance_ratio_)
    red_data = svd.transform(large_matrix)
    print(svd.explained_variance_ratio_.sum())
    print(svd.explained_variance_)

    # seed = np.random.RandomState(seed=3)
    # # bei precomputed muss man die dissimilarity vorher berechenen- irgendwie logisch
    # mds = manifold.MDS(n_components=2, max_iter=30, eps=1e-6, random_state=seed,
    #                    dissimilarity="euclidean", n_jobs=2, verbose=1)
    # pos = mds.fit(large_matrix).embedding_

    # with open("/home/tobias/Dokumente/saved_context_vectors/clob_crown/d100k3SVD50.pkl", 'wb') as output:
    #     pickle.dump(red_data, output)

    """
    fig = plt.figure(1)
    ax = plt.axes([0., 0., 1., 1.])
    s = 100
    plt.scatter(pos[:, 0], pos[:, 1], color='navy', s=s, lw=0,
                label='True Position')
    plt.show()
    """
    #rim = RIModel(dim, k)
    #rim.load_model_from_file('/home/tobias/Dokumente/saved_context_vectors/d10k3accu.pkl')
    #to_tree(rim=rim)
    """
    keys = [key for key in rim.ContextVectors.keys()]
    with open("/home/tobias/tree.pkl", 'rb') as inputFile:
        kdt = pickle.load(inputFile)

    test = "men"
    # bis jetzt nur der nächste nachbar
    dist, indices = kdt.query(rim.ContextVectors[test].toarray(), k=10)
    #index = int(ind)
    #print(len(keys), ind,keys[index])
    for ind in indices[0]:
        print(keys[int(ind)])
    """

    #merge_dicts(rmi.ContextVectors, rim.ContextVectors)

    # filename = "/home/tobias/Dokumente/pyworkspace/rindex/testdata/stateofunion.txt"
    # analyze_file_by_context(filename=filename,rmi=rmi,contextSize=2)
    # rmi.write_model_to_file("state_of_union")
    # bag of words?
    #rim.is_similar_to(word="men", thres=0.9, count=100)
    # line = 50*"-"
    # print(line)
    # rmi.reduce_dimensions(newDim =50)
    # rmi.write_model_to_file("sou")

    # for key in rmi.ContextVectors.keys():
    #     print(key)

    # vals = rmi.ContextVectors.values()
    # # Row-based linked list sparse matrix
    # largeMatrix = sp.lil_matrix((len(vals),dim))    
    # i = 0
    # for val in vals:
    #     largeMatrix[i] = val.transpose()
    #     i += 1
    # print(largeMatrix.shape[1])
    # targetSize = johnson_lindenstrauss_min_dim(largeMatrix.shape[1],0.4)
    # print(targetSize)
    # bei steigender wortanzahl lässt sich da sicher was rausholen
    # sparse = SparseRandomProjection(n_components = 10)
    # target = sparse.fit_transform(largeMatrix)
    # for t in target:
    #     print(t, end="\n\n")
    
    # filename = '/home/tobias/Dokumente/pyworkspace/rindex/testdata/stateofunion.txt'

    # dim = 1500
    # k = 3 # number of 1 and -1 
    # rmi = RIModel(dim,k)    
    # analyzeFilebyContext(filename, rmi, 2)
    # rmi.writeModelToFile()
    # filename= "/home/tobias/Dokumente/saved_context_vectors/d1500k3.pkl"
    # rmi.loadModelFromFile(filename)

    # rmi.ContextVectors = cv
    # rmi.reduceDimensions(100)
    # writeModelToFile(rmi.ContextVectors, 'c2_ContextVectors_reduced.pkl')
    # rmi.index_memory = im
    
    # rmi.isSimilarTo(word = "june", thres = 0.8, count = 100)
    # for key in rmi.ContextVectors.keys():
    #     print(key)

if __name__ == '__main__':
    main()

