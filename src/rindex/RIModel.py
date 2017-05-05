import itertools
import scipy.sparse as sp
from scipy import spatial
from scipy.stats import entropy

import numpy as np
from numpy.linalg import norm
import random

import pickle
# from rindex
import IndexVectorSciPy as IndexVec



"""
As far as I know pickle saves only class objects and not
complete objects. We could of course call it as a method of RIModel.
"""
def writeModelToFile(rmi, filename):
    with open(filename, 'wb') as output:
        pickle.dump(rmi, output)
def loadModelFromFile(filename):
    with open(filename, 'rb') as inputFile:
        return(pickle.load(inputFile))
fix_vec = lambda input:[(number,number+0.0001)[number == 0.0] for
                        number in input]




class RIModel:
    """RIModel - Random IndexVector Model
    
    """
    ContextVectors = {}
    iv = IndexVec.IndexVector
    dim = 0
    k = 0
    index_memory = {}

    def __init__(self, dim=0, k=0):
        self.dim = dim
        self.k = k
        self.iv = IndexVec.IndexVector(dim, k)



    def addDocument(self,context=[]):
        """     Takes the Context array as the context of its entries and each element of the array as word
        """
        for word in context:
            if word in self.ContextVectors.keys():
                pass
            else:
                                self.ContextVectors[word] = sp.coo_matrix((self.dim, 1))
            self.ContextVectors[word] += self.iv.createIndexVectorFromContext(word)


    def addContext(self, context = [], index = 0, mask = None):
        """Add a self defined Context for a specifix word with index , possibly with a weight mask
           default: index = 0 ( the first word in the array ) , if no
        mask given all contexts are weighted 1
        :param context[]:
        :param index:
        :param mask:
        """
        if not mask:
            mask = [1] * len(context)
        # für jedes word im Kontext wird dessen IndexVector auf den Kontextvektor
        # addiert. Der Vektor bezieht sich im default-Fall auf das 1. Wort.
        #
        # - create Hash for context
        # - seed
        # - addiere den erzeugten IndexVector aus dem Kontext
        #   auf den ContextVector des festlegen Wortes
        # -> das mit dem weight geht dann nicht.
        ## maske kann in der funktion darüber implementiert werden

        if context[index] not in self.ContextVectors.keys():
            self.ContextVectors[context[index]] = sp.coo_matrix((self.dim, 1))
    
        self.ContextVectors[context[index]] = self.iv.createIndexVectorFromContext(context)
            
            
        # for word, weight in zip(context, mask):
        
        #     if word == context[index]:
        #         continue
        #     if word not in self.index_memory.keys():
        #         ## schritt kann entfallen
        #                         self.index_memory[word] = self.iv.createIndexVectorFromContext([word])
        #     if context[index] not in self.ContextVectors.keys():
        #                         self.ContextVectors[context[index]] = sp.coo_matrix((self.dim, 1))
        #     self.ContextVectors[context[index]] += self.index_memory[word] * weight



    def getSimilarityCos(self, word1, word2):
         """    Calculate the Cosine-Similarity between the two elements word1 and word2
         word1, word2 must occur in the Model
         :param word1:
         :param word2:
         :return:
             """
         ## 1- ist bei mir so, kann man aber auch einfach ändern.
         return(1-spatial.distance.cosine(self.ContextVectors[word1].toarray(),self.ContextVectors[word2].toarray()))
    
    def getJSD(self, word1, word2):
        ## Das is irgenwie Käse. Außerdem bekomme ich grundsätzlich
        ## inf be entropy heraus.

        P = self.ContextVectors[word1].toarray().transpose()[0]
        Q = self.ContextVectors[word2].toarray().transpose()[0]

        ## Wie bekomme ich eine distribution von P,Q?
        _P = fix_vec(P / norm(P, ord=1))
        _Q = fix_vec(Q / norm(Q, ord=1))

        #_M = 0.5 * (_P + _Q)
        _M = (np.asarray(_P)+np.asarray(_Q)) * 0.5


        print(entropy(_Q,_M))
        # print array with seps
        #print(','.join(map(str,_M)))
        
        ## sqrt for distance, 1- for simularity
        return (0.5 * (entropy(_P, _M) + entropy(_Q, _M)))

     
    def isSimilarTo(self, word = "", thres = 0.9,count = 10):
        """ Returns words with the least distance to the given word.
        The combination of threshold and count can be used e.g. for 
        testing to get a small amount of words (and stop after that).
        
        :param word:
        :param thres:
        :param count
        :return list of words
        """
        if len(word) == 0:
               return
        try:
           # check if ContextVector exists
           self.ContextVectors[word]
        except:
            print("Word not in context")
            return
        i = 0
        for key in self.ContextVectors.keys():                
            if key != word:
                sim = self.getSimilarityCos(word,key)                

                if i == count:
                    break
                if sim > thres and i < count:
                    print(key,sim)
                    i += 1


                    

    def reduceDimensions(self, newDim = 100):
        """ Converts contextVectors to large matrix 
        and multiplies it with a random matrix to reduce dim.
        :param newDim:
        """
        ## generate a random matrix, self.k could be different.
        row = np.array([random.randint(0, self.dim-1) for x in range(self.k)])
        col = np.array([0 for x in range(self.k)])
        values = np.array([random.choice([-1, 1]) for x in range(self.k)])
        randMatrix = sp.coo_matrix((values, (row,col)), (self.dim, newDim))
        ## convert values of context-vectors to one large matrix (?)
        vals = self.ContextVectors.values()
        largeMatrix = sp.csr_matrix((len(vals),self.dim))    
        i = 0
        for val in vals:
            """
            SparseEfficiencyWarning: Changing the sparsity structure of a csr_matrix is expensive. lil_matrix is more efficient.
  SparseEfficiencyWarning)
            """
            largeMatrix[i] = val.transpose()
            i += 1
        reducedMatrix = largeMatrix * randMatrix
        ## back to dict
        cKeys = self.ContextVectors.keys()
        i = 0
        for key in cKeys:
            self.ContextVectors[key] = largeMatrix[i].transpose()
            i += 1

                        
    def mostSimilar(self,count= 10, file = None ):
        """ Compare all words in model. (Takes long Time)
        Isn't that one reason why we need dim-reduction?
        :param count:
        :param file:
        :return:
        """
        simDic = []
        keys = self.ContextVectors.keys()
        tuples = list(itertools.combinations(keys, 2))
        print("Comparing everything...")

        i = 0
        size = len(tuples)
        for pair in tuples:
            i+=1

            print("\r%f %%" % (100*i/size), end="")
            simDic.append([pair[0], pair[1], self.getSimilarityCos(pair[0], pair[1])])
        print("Sorting...")
        simDic.sort(key=lambda x: x[2], reverse=True)
        for x in range(count):
            print(x, ":\t", simDic[x])
        if file:
            with open(file, 'w') as out:
                for triple in simDic:
                    out.write(triple[0] + "\t" + triple[1] + "\t" + str(triple[2]) + "\n")


                    
def main():
    """Main function if the Module gets executed"""
    r = RIModel(100, 3)
    r.addContext(["hello", "world", "damn"])
    ## gleiches doc + randomVektor des Dokuments (?)
    r.addContext(["hello", "world", "damn"], index=0, mask=[0, 0.5, 0.5])
    r.addContext(["hello", "world", "example"], index=0, mask=[0, 0.5, 0.5])
    r.addContext(["hello", "world", "damn"], index=0, mask=[0, 0.5, 0.5])
    r.addContext(["hello", "world", "example"], index=0, mask=[0, 0.5, 0.5])
    r.addContext(["hello", "world", "example"], index=0, mask=[0, 0.5, 0.5])
    r.addContext(["hello", "damn", "nice"], index=0, mask=[0, 0.5, 0.5])

    r.addContext(["the", "damn", "example"], index=0, mask=[0, 0.5, 0.5])
    r.addContext(["the", "world", "example"], index=0, mask=[0, 0.5, 0.5])
    r.addContext(["the", "world", "example"], index=0, mask=[0, 0.5, 0.5])
    r.addContext(["the", "world", "example"], index=0, mask=[0, 0.5, 0.5])
    r.addContext(["the", "world", "nice"], index=0, mask=[0, 0.5, 0.5])
    r.addContext(["the", "world", "nice"], index=0, mask=[0, 0.5, 0.5])

    r.addContext(["the", "world", "nice"], index=0, mask=[0, 0.5, 0.5])    
    r.addContext(["parks", "are", "shitty"], index=0, mask=[0, 0.5, 0.5])
    

  #   vals = r.ContextVectors.values()
  #   largeMatrix = sp.csr_matrix((len(vals),50))    
  #   i = 0
  #   for val in vals:
  #       """
  #       SparseEfficiencyWarning: Changing the sparsity structure of a csr_matrix is expensive. lil_matrix is more efficient.
  # SparseEfficiencyWarning)
  #       """
  #       largeMatrix[i] = val.transpose()
  #       i += 1
                   
  #   print(largeMatrix)

    
    # for key in r.index_memory.keys():
    #     print(key, ":\t", r.index_memory[key].toarray())

    #r.getJSD("hello", "parks")
    r.isSimilarTo(word = "the", thres = 0.1,count = 10)
    # writeModelToFile(r, 'testModel.pkl')
    #rmi = loadModelFromFile('testModel.pkl')
    #print(rmi.getSimilarityCos("hello","the"))
    
        
if __name__ == "__main__":
    main()
