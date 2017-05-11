import hashlib

import numpy as np
import scipy.sparse as sp

# :8 because seeding allows only 'small' values
# looks like we don't nee this. have to check wether hash-function is the same
# on every system!
string_to_hash = lambda input: int(hashlib.md5(input.encode('utf-8')).hexdigest()[:4], 16)


class IndexVector:
    vec = sp.coo_matrix
    randomIndexDB = {}
    dim = 10  # default
    # count of  -1,1
    n = 2  # default
    prob_minus = prob_plus = prob_zero = 0.0

    def __init__(self, dim, n):
        self.dim = dim
        self.n = n
        self.set_prob()

    def set_prob(self):
        # only needed for tobi's 'hashing'
        self.prob_zero = (self.dim - self.n) / self.dim
        self.prob_minus = self.prob_plus = (self.n / self.dim) / 2

    def create_index_vector_from_context(self, context=[]):
        """
            Creates an Index-Vector given a context array with words.
            __Notes__:
            seeding with same array -> same value
            could be done in a similar way janos did in createIndexVectorRandom
            :param context[]:
        """
        ## summieren des "gekappten" hashs
        ## wird trotzdem zu groß. 'ein string' ist auch keine lösung @todo
        sumOfSeeds = 0.0000
        for word in context:
            sumOfSeeds += string_to_hash(word)

        np.random.seed(int(sumOfSeeds))
        self.vec = sp.csr_matrix(np.random.choice([0, 1, -1], size=(self.dim, 1),
                                                  p=[self.prob_zero, self.prob_plus, self.prob_minus]))
        return self.vec

    def create_index_vector_random(self, context=[]):
        ## is this the place where you choose the indexes
        ## or rather how do you save the values?

        ## This works because random generates the same sequence of values after
        ## the same seed. After all, I like this method better because we have full
        ## control about the count of -1 and 1s.
        ## -> might be faster, too
        np.random.seed(string_to_hash(context[0]))  ##

        row = np.array([np.random.randint(0, self.dim - 1) for x in range(self.n)])
        col = np.array([0 for x in range(self.n)])

        values = np.array([np.random.choice([-1, 1]) for x in range(self.n)])
        self.vec = sp.coo_matrix((values, (row, col)), (self.dim, 1))
        return self.vec

    def set(self, dimension=0, n=0):
        self.dim = dimension
        self.n = n




def main():
    iv = IndexVector(100, 3)
    testSentence = ["I", "like", "Parks", "and", "Recreation"]

    for word in testSentence + testSentence:
        iv.create_index_vector_from_context([word])
        print(word, iv.vec)
        print()

        # for i in range(10):
        #     iv.createIndexVectorRandom([testSentence[0]])
        #     print(iv.vec)
        #     print()
        # testHash(testSentence)

        # i.createIndexVectorRandom()
        # print(i.vec)
        # print()

        # i.createIndexVectorFromContext(['hello','world'])
        # print(i.vec)
        # print()
        # i.createIndexVectorFromContext(['world','world'])
        # print(i.vec)
        # print()
        # i.createIndexVectorFromContext(['world','hello'])
        # print(i.vec)


if __name__ == "__main__":
    main()
