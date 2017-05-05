
from nltk.tokenize import word_tokenize as w_tokenize
from nltk.tokenize import sent_tokenize as s_tokenize
import re
import time
import os

import IndexVectorSciPy as IndexVec
from RIModel import RIModel
## im Moment nur als Funktionen vorhanden.
from RIModel import writeModelToFile
from RIModel import loadModelFromFile



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

        
# def analyzeFilebyContext(filename, rmi, contextSize = 2):

#     with open(filename) as fin:
#         text = fin.read()
#     content = w_tokenize(text)

#     start = time.time()
#     size = len(content)
#     for i in range(len(content)):
#         print("\r%f %%" % (100*i/size), end="")        
#         word = content[i] ## scheine ich zu vergessen...
#         context = []
#         for j in range(1,contextSize+1):
#             try:
#                 context.append(content[j+i])
#                 i += 1 ## without 'window'
#             except:
#                 pass
#         try:
#             rmi.addContext(context, index = 0)
#         except:
#             print(context)
#     print(time.time()-start)

#     writeModelToFile(rmi.ContextVectors, 'c2_ContextVectors.pkl')  
#     writeModelToFile(rmi.index_memory, 'c2_index_memory.pkl')


def main():

    filename = '/home/tobias/Dokumente/pyworkspace/rindex/testdata/stateofunion.txt'

    dim = 1500
    k = 10 # number of 1 and -1 
    rmi = RIModel(dim,k)    
    #analyzeFilebySentence(filename, rmi)
    filename= "/home/tobias/Dokumente/saved_context_vectors/d1500k10.pkl"
    rmi.loadModelFromFile(filename)
    # cv = loadModelFromFile('c2_ContextVectors.pkl')
    
    # rmi.ContextVectors = cv
    # # rmi.reduceDimensions(100)
    # #writeModelToFile(rmi.ContextVectors, 'c2_ContextVectors_reduced.pkl')  
    # #rmi.index_memory = im
    
    rmi.isSimilarTo(word = "president", thres = 0.4, count = 100)





    
if __name__ == '__main__':
    main()
    

