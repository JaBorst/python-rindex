
from nltk.tokenize import word_tokenize as w_tokenize
from nltk.tokenize import sent_tokenize as s_tokenize
import re
import os
import pickle

from RIModel import RIModel
import numpy as np


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
    list_of_files = []
    i = 0
    for (dirpath, dirnames, filenames) in os.walk(path):
        for filename in filenames:
            if filename.endswith('.txt'): 
                list_of_files.append(os.sep.join([dirpath, filename]))
    return list_of_files


# @todo: anderer name
def analyze_files_of_folder(path="", contextSize = 2, ext = ""):
    # walks through folder and builds a model
    # ideas:
    # - add a random (random) vector for each document
    #   to each context in this document
    # - build the models incremental, i.e. one model for
    #   each file, then add them up -> is this possible?
    # -> at the moment it's just one model, also ok
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
    # as the name suggests: if we have more than one dict/model
    # this is an incremental function to merge them
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
    from sklearn.neighbors import KDTree
    # should be done !only! with reduced data
    # to enhence search
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
    rmi = RIModel(dim, k)
    file_source = "/home/tobias/Dokumente/testdata/stateofunion.txt"
    folder_source = "/home/tobias/Downloads/OANC/data/written_1"

    #analyze_files_of_folder(path=folder_source,contextSize=2,ext="written_1")
    #analyze_file_by_context(filename=file_source,rmi=rmi, contextSize=5)
    #rmi.write_model_to_file("sou_5")
    rmi.load_model_from_file('/home/tobias/Dokumente/saved_context_vectors/d50k3svd_accu.pkl')

    rmi.is_similar_to(word="girl", thres=0.9, count=100)


    # # bei precomputed muss man die dissimilarity vorher berechenen- irgendwie logisch
    # seed = np.random.RandomState(seed=3)
    # mds = manifold.MDS(n_components=2, max_iter=30, eps=1e-6, random_state=seed,
    #                    dissimilarity="euclidean", n_jobs=2, verbose=1)
    # pos = mds.fit(large_matrix).embedding_

    #with open("/home/tobias/Dokumente/saved_context_vectors/clob_crown/d100k3SVD50.pkl", 'wb') as fout:
    #    pickle.dump(rmi.ContextVectors, fout)



    """
    fig = plt.figure(1)
    ax = plt.axes([0., 0., 1., 1.])
    s = 100
    plt.scatter(pos[:, 0], pos[:, 1], color='navy', s=s, lw=0,
                label='True Position')
    plt.show()

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
