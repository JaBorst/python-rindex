
from nltk.tokenize import word_tokenize as w_tokenize
from nltk.tokenize import sent_tokenize as s_tokenize
import re
import time
import os

import IndexVectorSciPy as IndexVec
from RIModel import RIModel

## should not be here
from sklearn.random_projection import johnson_lindenstrauss_min_dim
from sklearn.random_projection import SparseRandomProjection
import scipy.sparse as sp

def clean_word_seq(context = []):
    ## some trimming
    if len(context) > 1:
        return([word.lower() for word in context if re.search(r'[a-zA-Z]', word) is not None])
    elif len(context) == 0:
        return
    elif len(context) == 1:
        if re.search(r'[a-zA-Z]', context[0]) is not None:
            return context[0].lower()
        else:
            return

def analyzeFilebySentence(filename, rmi):
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
        rmi.addContext(w_tokenize(sentence), index = 0)
        i += 1

    rmi.writeModelToFile()

        
def analyzeFilebyContext(filename, rmi, contextSize = 2):

    with open(filename) as fin:
        text = fin.read()
    content = s_tokenize(text)

    size = len(content)

    i = 0
    doc_iv = rmi.iv.createIndexVectorFromContext(filename)
    
    for sentence in content:

        # if i%100 == 0:
        #     print("\r%f %%" % (100*i/size), end="")

        ## man kann natürlich auch hier das 1. wort rausnehmen
        ## für jedes wort
        
        sent = clean_word_seq(w_tokenize(sentence))
        try:
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
                        rmi.addContext(context, index = 0)
                        ## add something for each doc extrax
                        #rmi.ContextVectors[context[0]] += doc_iv
                    except:
                        pass
        except:
            pass            
        i += 1
        
        
        


def getPathsOfFiles(path):
    list_of_files = {}
    for (dirpath, dirnames, filenames) in os.walk(path):
        for filename in filenames:
            if filename.endswith('.txt'): 
                list_of_files[filename] = os.sep.join([dirpath, filename])
    return list_of_files

## @todo: anderer name
def analyzeFilesOfFolder(path):
    ## walks through folder and builds a model
    ## ideas:
    ## - add a random (random) vector for each document
    ##   to each context in this document
    ## - build the models incremental, i.e. one model for
    ##   each file, then add them up -> is this possible?
    ## -> at the moment it's just one model
    
    files = getPathsOfFiles(path)
    rmi = RIModel(dim = 1000, k = 3)
    i = 0
    size = len(files)
    for filename in files.values():
        if i%100 == 0:
            print("\r%f %%" % (100*i/size), end="")

        analyzeFilebyContext(filename, rmi, contextSize = 2)
        if i%1000 == 0:
            ## maybe just for testing save the model once in a
            ## while: could save some nerves.
            rmi.writeModelToFile()
        if i > 10:
            break
        i += 1
    rmi.writeModelToFile()  


def main():
    path= "/home/tobias/Dokumente/pyworkspace/rindex/testdata/Newspapers/CLOB_RAW"
    #analyzeFilesOfFolder(path)
    dim = 1000
    k = 3
    rmi = RIModel(dim ,k)
    rmi.loadModelFromFile('/home/tobias/Dokumente/saved_context_vectors/d1000k3.pkl')
    # rmi.isSimilarTo(word = "father", thres = 0.7, count =100)
    # for key in rmi.ContextVectors.keys():
    #     print(key)

    vals = rmi.ContextVectors.values()
    # Row-based linked list sparse matrix
    largeMatrix = sp.lil_matrix((len(vals),dim))    
    i = 0
    for val in vals:
        """
            SparseEfficiencyWarning: Changing the sparsity structure of a csr_matrix is expensive. lil_matrix is more efficient.
  SparseEfficiencyWarning)
        """
        largeMatrix[i] = val.transpose()
        i += 1
    print(largeMatrix.shape[1])
    targetSize = johnson_lindenstrauss_min_dim(largeMatrix.shape[1],0.4)
    print(targetSize)
    ## bei steigender wortanzahl lässt sich da sicher was rausholen
    sparse = SparseRandomProjection(n_components = 10)
    target = sparse.fit_transform(largeMatrix)
    # for t in target:
    #     print(t, end="\n\n")
    
    # filename = '/home/tobias/Dokumente/pyworkspace/rindex/testdata/stateofunion.txt'

    # dim = 1500
    # k = 3 # number of 1 and -1 
    # rmi = RIModel(dim,k)    
    # #analyzeFilebyContext(filename, rmi, 2)
    ## rmi.writeModelToFile()  
    # filename= "/home/tobias/Dokumente/saved_context_vectors/d1500k3.pkl"
    # rmi.loadModelFromFile(filename)

    
    # # rmi.ContextVectors = cv
    # # # rmi.reduceDimensions(100)
    # # #writeModelToFile(rmi.ContextVectors, 'c2_ContextVectors_reduced.pkl')  
    # # #rmi.index_memory = im
    
    # rmi.isSimilarTo(word = "june", thres = 0.8, count = 100)
    # # for key in rmi.ContextVectors.keys():
    # #     print(key)




    
if __name__ == '__main__':
    main()
    
