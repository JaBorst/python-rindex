
from nltk.tokenize import word_tokenize as w_tokenize
from nltk.tokenize import sent_tokenize as s_tokenize
import re
import time
import os
import multiprocessing

import IndexVectorSciPy as IndexVec
from RIModel import RIModel





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

    ## file-decoding festlegen
    try:
        with open(filename,'r' ,encoding='utf8') as fin:
            text = fin.read()
    except:
        print("error with ",filename)
        return
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
    list_of_files = []
    i = 0
    for (dirpath, dirnames, filenames) in os.walk(path):
        for filename in filenames:
            if filename.endswith('.txt'): 
                list_of_files.append(os.sep.join([dirpath, filename]))
    return list_of_files

## @todo: anderer name
def analyzeFilesOfFolder(path, contextSize = 2):
    ## walks through folder and builds a model
    ## ideas:
    ## - add a random (random) vector for each document
    ##   to each context in this document
    ## - build the models incremental, i.e. one model for
    ##   each file, then add them up -> is this possible?
    ## -> at the moment it's just one model

    files = getPathsOfFiles(path)
    rmi = RIModel(dim = 1500, k = 3)
    
    # procs = 3
    # size = len(files)
    # ## möglich, dass das in einem desaster endet. evtl je file ein rimodel
    # ## kann eigentlich gar nicht gehen. 
    # for f_i in range(0,len(files),procs):
    #     if f_i%10 == 0:
    #         print("\r%f %%" % (100*f_i/size), end="")
    #     ## schnappt sich immer ein paar files (je nach proc anzahl)j
    #     jobs = []
    #     for i in range(procs):
    #         process = multiprocessing.Process(target=analyzeFilebyContext,
    #     	                                      args=(files[i], rmi, i,contextSize))
    #         jobs.append(process)
    #     for j in jobs:
    #         j.start()
    #     for j in jobs:
    #         j.join()
    #     # if f_i > 3:
    #     #     break
    #     if f_i%50 == 0:
    #         print("save at ", f_i)
    #         rmi.writeModelToFile()
        
    i = 0
    size = len(files)
    print(size)
    for filename in files:
        if i%10 == 0:
            print("\r%f %%" % (100*i/size), end="")

        analyzeFilebyContext(filename, rmi, contextSize = 2)
        if i%50 == 0:
            ## maybe just for testing save the model once in a
            ## while: could save some nerves.
            rmi.writeModelToFile()
        i += 1
    rmi.writeModelToFile()  


def main():
    path= "/home/tobias/Dokumente/pyworkspace/rindex/testdata/Newspapers/Crown_RAW"
    #analyzeFilesOfFolder(path, contextSize=2)
    dim = 1500
    k = 3
    rmi = RIModel(dim ,k)
    rmi.loadModelFromFile('/home/tobias/Dokumente/saved_context_vectors/d10k3.pkl')
    rmi.isSimilarTo(word = "london", thres = 0.95, count =100)
    line = 50*"-"
    print(line)
    #    rmi.reduceDimensions(newDim =10)# macht noch nix

    # for key in rmi.ContextVectors.keys():
    #     print(key)
    #rmi.writeModelToFile()

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
    # ## bei steigender wortanzahl lässt sich da sicher was rausholen
    # sparse = SparseRandomProjection(n_components = 10)
    # target = sparse.fit_transform(largeMatrix)
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
    

