
from nltk.tokenize import word_tokenize as w_tokenize
from nltk.tokenize import sent_tokenize as s_tokenize
import re
import time
import os

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

    with open(filename) as fin:
        text = fin.read()
    content = s_tokenize(text)

    size = len(content)

    i = 0

    for sentence in content:

        if i%100 == 0:
            print("\r%f %%" % (100*i/size), end="")

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
                    except:
                        pass
        except:
            pass            
        i += 1
        
        
        
    rmi.writeModelToFile()  



def main():

    filename = '/home/tobias/Dokumente/pyworkspace/rindex/testdata/stateofunion.txt'

    dim = 1500
    k = 3 # number of 1 and -1 
    rmi = RIModel(dim,k)    
    #analyzeFilebyContext(filename, rmi, 2)
    filename= "/home/tobias/Dokumente/saved_context_vectors/d1500k3.pkl"
    rmi.loadModelFromFile(filename)

    
    # rmi.ContextVectors = cv
    # # rmi.reduceDimensions(100)
    # #writeModelToFile(rmi.ContextVectors, 'c2_ContextVectors_reduced.pkl')  
    # #rmi.index_memory = im
    
    rmi.isSimilarTo(word = "men", thres = 0.9, count = 100)
    # for key in rmi.ContextVectors.keys():
    #     print(key)




    
if __name__ == '__main__':
    main()
    

