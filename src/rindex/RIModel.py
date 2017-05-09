import itertools
import scipy.sparse as sp
from scipy import spatial

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
    #index_memory = {}
    is_sparse = True

    def __init__(self, dim=0, k=0):
        self.dim = dim
        self.k = k
        self.iv = IndexVec.IndexVector(dim, k)
        self.file_path = "/home/tobias/Dokumente/saved_context_vectors/"

    def write_model_to_file(self, add_text = ""):
        filename = self.file_path+ "d"+str(self.dim)+"k"+str(self.k)+add_text+".pkl"
        print("write model to ",filename)
        with open(filename, 'wb') as output:
            pickle.dump(self.ContextVectors, output)

    def load_model_from_file(self, filename):
        with open(filename, 'rb') as inputFile:
            self.ContextVectors = pickle.load(inputFile)
        # get any element and determine if it's sparse or not
        some_element = self.ContextVectors[next(iter(self.ContextVectors))]
        if sp.issparse(some_element):
            self.is_sparse = True
        else:
            self.is_sparse = False

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
         :return: cosinus-distance
             """
         # 1- ist bei mir so, kann man aber auch einfach ändern.
        if self.is_sparse:
            return 1-spatial.distance.cosine(self.ContextVectors[word1].toarray(),
                                             self.ContextVectors[word2].toarray())
        else:
            return 1 - spatial.distance.cosine(self.ContextVectors[word1],
                                               self.ContextVectors[word2])
    

    def is_similar_to(self, word ="", thres = 0.9, count = 10):
        """ Returns words with the least distance to the given word.
        The combination of threshold and count can be used e.g. for 
        testing to get a small amount of words (and stop after that).
        :param word:
        :param thres:
        :param count
        :return list of similar words
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
        sims = {}
        for key in self.ContextVectors.keys():
            if key != word:
                sim = self.get_similarity_cos(word, key)
                sims[key] = sim
                if sim > thres and i < count:
                    print(key, sim)
                    i += 1

        # hier geht was schief
        # if i < count:
        #     print("\n\n {0} most similar words".format(count) )
        #     best_n = dict(sorted(sims.items(), key=operator.itemgetter(1), reverse=True)[:count])
        #     for word in best_n.keys():
        #         print(word,"\t\t", best_n[word])

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

    def reduce_dimensions(self,method= "random_projection", target_size=100):
        # kann ich sicher sein, dass der richtige vector das
        # richtige key-wort bekommt?
        # ->ja, da ich die Vektoren beim Erzeugen der Matrix in der
        # Reihenfolge abgespeichert habe.

        keys = self.ContextVectors.keys()
        # Row-based linked list sparse matrix
        target_martix = sp.lil_matrix((len(keys),self.dim))
        i = 0
        for key in keys:
            target_martix[i] = self.ContextVectors[key].transpose()
            i += 1

        print("Reduce ",target_martix.shape[1], " to ", target_size)
        self.ContextVectors = {} # reset dicct
        if method == "random_projection":
            """
                SPARSE_RANDOM_PROJECTION
            """
            print("using SparseRandomProjection...")
            from sklearn.random_projection import SparseRandomProjection
            sparse = SparseRandomProjection(n_components = target_martix)
            red_data = sparse.fit_transform(target_martix)
            i = 0
            for key in keys:
                self.ContextVectors[key] = red_data[i][0]
                i += 1
            self.is_sparse = True
        elif method == "truncated_svd":
            """
                TRUNCATED_SVD
            """
            from sklearn.decomposition import TruncatedSVD
            print("using TruncatedSVD...")
            # 50  seems to be a good value (maybe les)
            svd = TruncatedSVD(n_components=target_size, n_iter=10, random_state=42)
            red_data = svd.fit_transform(target_martix)
            print("sd-sum is:\t", svd.explained_variance_ratio_.sum())
            i = 0
            for key in keys:
                self.ContextVectors[key] = red_data[i]
                i += 1
            self.is_sparse = False
        elif method == "tsne":
            print("not yet implemented")

        self.dim = target_size


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
