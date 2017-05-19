import sys
sys.path.append('../')
import multiprocessing

import pickle
from t_helper import get_paths_of_files
from t_helper import clean_word_seq
from nltk.tokenize import word_tokenize as w_tokenize
from rindex import *


def bm_on_file(filename, f_number, context_size=2, dim=1500, k=5):
    """
    
    :param filename: 
    :param f_number: 
    :param context_size: 
    :param dim: 
    :param k: 
    :return: 
    """
    with open(filename, 'r', encoding="utf-8") as fin:
        sents = fin.readlines()
    print(filename)
    size = len(sents)

    rmi = RIModel.RIModel(dim, k)
    # rmi.is_sparse = True

    for i, sent in zip(range(len(sents)), sents):
        if i % 100 == 0:
            print("\r%f %%" % (100 * i / size), end="")
        sent = clean_word_seq(w_tokenize(sent))
        for j in range(len(sent)):
            context = []
            for k in range(j, j + context_size):
                try:
                    context.append(sent[k])
                except:
                    pass
            if len(context):
                try:
                    rmi.add_context(context, index=1)
                except:
                    pass
                    # break
    fout = "/home/tobias/Dokumente/saved_context_vectors/paratest/small_word_siml"+str(f_number)+".model"
    rmi.write_model_to_file(fout)
    # break


def merge_model(rim1, rim2):
    """
    as the name suggests: if we have more than one dict/model
    this is an incremental function to merge them
    :param path
    :return:
    """
    if rim1.dim != rim2.dim or rim1.k != rim2.k:
        print("dim,k not matching")
        return

    for key in rim1.ContextVectors.keys():
        if key in rim2.ContextVectors.keys():
            rim1.ContextVectors[key] = rim1.ContextVectors[key] + rim2.ContextVectors[key]
        else:
            rim1.ContextVectors[key] = rim1.ContextVectors[key]
    for key in rim2.ContextVectors.keys():
        if key not in rim1.ContextVectors.keys():
            rim1.ContextVectors[key] = rim2.ContextVectors[key]


def bwhole(path="", target_file=""):
    """
    
    :param path: 
    :param target_file: 
    :return: 
    """
    model_files = get_paths_of_files(path, filetype=".model")
    rim1 = RIModel.RIModel(dim=1, k=1)# dummy
    rim2 = RIModel.RIModel(dim=1, k=1)# dummy

    print("start merging.")
    rim1.load_model_from_file(model_files[0])
    for i in range(1,len(model_files)):
        if model_files[i] == target_file or model_files[i] == "/home/tobias/Dokumente/saved_context_vectors/paratest/merge.model":
            continue
        rim2.load_model_from_file(model_files[i])
        merge_model(rim1=rim1,rim2=rim2)

    rim1.write_model_to_file("/home/tobias/Dokumente/saved_context_vectors/paratest/merge.model")
    rim1.reduce_dimensions(method="truncated_svd", target_size=50)
    rim1.write_model_to_file(target_file)

    keys, kdt = rim1.to_tree(method="minkowski", leaf_size=50)
    with open("/home/tobias/Dokumente/saved_context_vectors/paratest/word_sim.tree", 'wb') as output:
        pickle.dump(kdt, output)
    with open("/home/tobias/Dokumente/saved_context_vectors/paratest/word_sim.keys", 'wb') as output:
        pickle.dump(keys, output)


def main():
    path = "/home/tobias/Dokumente/testdata/wortschatz_small"
    target_path= "/home/tobias/Dokumente/saved_context_vectors/paratest"
    files = get_paths_of_files(path, filetype=".txt")
    procs = 3
    k = 2
    dim = 1500
    context_size = 2
    for i in range(0,len(files),procs):
        jobs = []
        for j in range(0, procs):
            if i+j < len(files):
                process = multiprocessing.Process(target=bm_on_file,
                                                  args=(files[i+j],i+j,context_size, dim, k))
            jobs.append(process)

        for j in jobs:
            j.start()
        # Ensure all of the processes have finished
        for j in jobs:
            j.join()
    # at the end merge models
    target_file = target_path + "/accu.model"
    bwhole(path=target_path,
           target_file=target_file)

if __name__ == "__main__":
    main()
    # rim1 = RIModel.RIModel(dim=1, k=1)# dummy
    # rim2 = RIModel.RIModel(dim=1, k=1)# dummy
    # rim1.load_model_from_file("/home/tobias/Dokumente/saved_context_vectors/d1500accu_2.model")
    # rim2.load_model_from_file("/home/tobias/Dokumente/saved_context_vectors/oanc/d1500k3written_1.pkl")
    # print(rim1.dim,rim2.dim)
    # merge_model(rim1,rim2)
    # #            rim2.load_model_from_file("/home/tobias/Dokumente/saved_context_vectors/clob_crown/d1500k3merge.pkl"))
    #rim1.write_model_to_file("/home/tobias/Dokumente/saved_context_vectors/d1500accu_2.model")
    # bwhole(path="/home/tobias/Dokumente/saved_context_vectors/paratest",
    #        target_file="/home/tobias/Dokumente/saved_context_vectors/paratest/accu")

