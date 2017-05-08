import itertools
import scipy.sparse as sp
from scipy import spatial

from sklearn.random_projection import SparseRandomProjection
from sklearn.random_projection import johnson_lindenstrauss_min_dim
import numpy as np
import pickle
#from rindex
import IndexVectorSciPy as IndexVec
import operator
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

    def write_model_to_file(self, add_text = ""):
        filepath = "/home/tobias/Dokumente/saved_context_vectors/"
        filename = filepath+ "d"+str(self.dim)+"k"+str(self.k)+add_text+".pkl"
        print(filename)
        with open(filename, 'wb') as output:
            pickle.dump(self.ContextVectors, output)

    def load_model_from_file(self, filename):
        with open(filename, 'rb') as inputFile:
            self.ContextVectors = pickle.load(inputFile)

    def add_document(self, context=[]):
        """     Takes the Context array as the context of its entries and each element of the array as word
        """
        for word in context:
            if word in self.ContextVectors.keys():
                pass
            else:
                                self.ContextVectors[word] = sp.coo_matrix((self.dim, 1))
            self.ContextVectors[word] += self.iv.create_index_vector_from_context(word)

    def add_context(self, context = [], index = 0, mask = None):
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

        ## maske kann in der funktion darüber implementiert werden
        ## frage: bezieht sich das word im context auf sich selbst?

        if context[index] not in self.ContextVectors.keys():
            self.ContextVectors[context[index]] = sp.coo_matrix((self.dim, 1))

        ## add everything but the word at index to the contextVector beeing
        ## created.
        rest = context[:index] + context[index+1:]
        self.ContextVectors[context[index]] += self.iv.create_index_vector_from_context(rest)

        # for word, weight in zip(context, mask):
        
        #     if word == context[index]:
        #         continue
        #     if word not in self.index_memory.keys():
        #         ## schritt kann entfallen
        #                         self.index_memory[word] = self.iv.createIndexVectorFromContext([word])
        #     if context[index] not in self.ContextVectors.keys():
        #                         self.ContextVectors[context[index]] = sp.coo_matrix((self.dim, 1))
        #     self.ContextVectors[context[index]] += self.index_memory[word] * weight

    def get_similarity_cos(self, word1, word2):
        """    Calculate the Cosine-Similarity between the two elements word1 and word2
         word1, word2 must occur in the Model
         :param word1:
         :param word2:
         :return:
             """
         # 1- ist bei mir so, kann man aber auch einfach ändern.
        return 1-spatial.distance.cosine(self.ContextVectors[word1].toarray(),self.ContextVectors[word2].toarray())
    
    # def getJSD(self, word1, word2):
    #     ## Das is irgenwie Käse. Außerdem bekomme ich grundsätzlich
    #     ## inf be entropy heraus.

    #     P = self.ContextVectors[word1].toarray().transpose()[0]
    #     Q = self.ContextVectors[word2].toarray().transpose()[0]

    #     ## Wie bekomme ich eine distribution von P,Q?
    #     _P = fix_vec(P / norm(P, ord=1))
    #     _Q = fix_vec(Q / norm(Q, ord=1))

    #     #_M = 0.5 * (_P + _Q)
    #     _M = (np.asarray(_P)+np.asarray(_Q)) * 0.5
    #     print(entropy(_Q,_M))
    #     # print array with seps
    #     #print(','.join(map(str,_M)))
        
    #     ## sqrt for distance, 1- for simularity
    #     return (0.5 * (entropy(_P, _M) + entropy(_Q, _M)))

    def is_similar_to(self, word ="", thres = 0.9, count = 10):
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
        max_sim = 0.0
        max_sim_word = ""

        sims = {}
        for key in self.ContextVectors.keys():
            if key != word:
                sim = self.get_similarity_cos(word, key)
                sims[key] = sim
                if sim > thres and i < count:
                    print(key, sim)
                    i+=1
        n = 10
        best_n = dict(sorted(sims.items(), key=operator.itemgetter(1), reverse=True)[:n])
        for word in best_n.keys():
            print(word,"\t\t", best_n[word])



    def reduce_dimensions(self, newDim = 100):
        """ Converts contextVectors to large matrix 
        and multiplies it with a random matrix to reduce dim.
        :param newDim:
        """

        # Row-based linked list sparse matrix
        keys = self.ContextVectors.keys()
        large_matrix = sp.lil_matrix((len(keys),self.dim))
        i = 0
        for key in keys:
            large_matrix[i] = self.ContextVectors[key].transpose()
            i += 1

        #targetSize = johnson_lindenstrauss_min_dim(largeMatrix.shape[1],0.1)
        target_size = newDim
        print("Reduce ",large_matrix.shape[1], " to ", target_size)
        
        sparse = SparseRandomProjection(n_components = target_size)
        target = sparse.fit_transform(large_matrix)

        self.ContextVectors = {}
        i = 0
        # kann ich sicher sein, dass der richtige vector das
        # richtige key-wort bekommt?
        # ->ja, da ich die Vektoren beim Erzeugen der Matrix in der
        # Reihenfolge abgespeichert habe.
        for key in keys:
            self.ContextVectors[key] = target[i][0]
            i += 1

        self.dim = newDim 

    def most_similar(self, count=10, file=None):
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
            simDic.append([pair[0], pair[1], self.get_similarity_cos(pair[0], pair[1])])
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
    dim = 1000
    k = 3
    r = RIModel(dim, k)
    r.add_context(["hello", "world", "damn"])

    r.add_context(["hello", "world", "damn"], index=0, mask=[0, 0.5, 0.5])
    r.add_context(["hello", "world", "example"], index=0, mask=[0, 0.5, 0.5])
    r.add_context(["hello", "world", "damn"], index=0, mask=[0, 0.5, 0.5])
    r.add_context(["hello", "world", "example"], index=0, mask=[0, 0.5, 0.5])
    r.add_context(["hello", "world", "example"], index=0, mask=[0, 0.5, 0.5])
    r.add_context(["hello", "damn", "nice"], index=0, mask=[0, 0.5, 0.5])

    r.add_context(["the", "damn", "example"], index=0, mask=[0, 0.5, 0.5])
    r.add_context(["the", "world", "example"], index=0, mask=[0, 0.5, 0.5])
    r.add_context(["the", "world", "example"], index=0, mask=[0, 0.5, 0.5])
    r.add_context(["the", "world", "example"], index=0, mask=[0, 0.5, 0.5])
    r.add_context(["the", "world", "nice"], index=0, mask=[0, 0.5, 0.5])
    r.add_context(["the", "world", "nice"], index=0, mask=[0, 0.5, 0.5])

    r.add_context(["the", "world", "nice"], index=0, mask=[0, 0.5, 0.5])
    r.add_context(["parks", "are", "shitty"], index=0, mask=[0, 0.5, 0.5])
    
    # r.writeModelToFile()
    # rmi = RIModel(dim, 3)
    # filename = "/home/tobias/Dokumente/saved_context_vectors/d100k3.pkl"
    # rmi.loadModelFromFile(filename)
    

    #r.getJSD("hello", "parks")
    r.is_similar_to(word ="hello", thres = 0.1, count = 10)


if __name__ == "__main__":
    main()
