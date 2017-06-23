import sys
sys.path.append('../')
import multiprocessing

import pickle
from t_helper import get_paths_of_files
from t_helper import clean_word_seq
from readTest import get_weight_vector
from nltk.tokenize import word_tokenize as w_tokenize
from rindex import *
<<<<<<< HEAD


def bm_on_file(filename, f_number, context_size=2, dim=1500, k=5):
=======
import os.path
import time

"""
            BUILD-MODEL-FUNCTIONS
"""
def bm_on_file(filename, f_number, context_size=2, dim=1500, k=3):
>>>>>>> b3fec38... freitag
    """
    
    :param filename: 
    :param f_number: 
    :param context_size: 
    :param dim: 
    :param k: 
    :return: 
    """
<<<<<<< HEAD
    with open(filename, 'r', encoding="utf-8") as fin:
        sents = fin.readlines()
    print(filename)
=======
    # only for 1 sent one line!
    # with open(filename, 'r', encoding="utf-8") as fin:
    #     sents = fin.readlines()

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
    index =1
    weights = get_weight_vector(context_size, target_index=index)

    sents = s_tokenize(text)
>>>>>>> b3fec38... freitag
    size = len(sents)

    rmi = RIModel.RIModel(dim, k)
    # rmi.is_sparse = True

    for i, sent in zip(range(len(sents)), sents):
<<<<<<< HEAD
        if i % 100 == 0:
            print("\r%f %%" % (100 * i / size), end="")
=======
        # if i % 100 == 0:
        #     print("\r%f %%" % (100 * i / size), end="")
>>>>>>> b3fec38... freitag
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
<<<<<<< HEAD
                    rmi.add_context(context, index=1)
                except:
                    pass
                    # break
    fout = "/home/tobias/Dokumente/saved_context_vectors/paratest/small_word_siml"+str(f_number)+".model"
=======
                    #rmi.add_context(context, index=0)
                    rmi.add_unit(context[index], context, weights=weights)
                except:
                    pass
    fout = "/home/tobias/Dokumente/models/working/"+str(f_number)+".model"
>>>>>>> b3fec38... freitag
    rmi.write_model_to_file(fout)


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


<<<<<<< HEAD
def bwhole(path="", target_file=""):
=======
def merge_models_in_folder(path="", target_file=""):
>>>>>>> b3fec38... freitag
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
<<<<<<< HEAD
=======

    size= len(model_files)
>>>>>>> b3fec38... freitag
    for i in range(1,len(model_files)):
        if model_files[i] == target_file or model_files[i] == "/home/tobias/Dokumente/saved_context_vectors/paratest/merge.model":
            continue
        rim2.load_model_from_file(model_files[i])
        merge_model(rim1=rim1,rim2=rim2)

    rim1.write_model_to_file("/home/tobias/Dokumente/saved_context_vectors/paratest/merge.model")
    rim1.reduce_dimensions(method="truncated_svd", target_size=50)
    rim1.write_model_to_file(target_file)

    keys, kdt = rim1.to_tree(method="minkowski", leaf_size=50)
    with open("/home/tobias/Dokumente/saved_context_vectors/word_sim.tree", 'wb') as output:
        pickle.dump(kdt, output)
    with open("/home/tobias/Dokumente/saved_context_vectors/word_sim.keys", 'wb') as output:
        pickle.dump(keys, output)

def split_text_file(filename="", chunks=1):
    """

    :param filename:
    :param chunks:
    :return:
    """
    with open(filename, 'r', encoding='utf-8') as fin:
        text = fin.readlines()
    lines_in_new_file = int(len(text) / chunks)
    index = 0
    for i in range(chunks):
        n_filename= os.path.split(filename)[0]+"/"+str(i)
        with open(n_filename, 'a') as fout:
            for j in range(lines_in_new_file):
                fout.write(text[index]+"\n")
                index+=1


def main():
<<<<<<< HEAD
    path = "/home/tobias/Dokumente/testdata/wortschatz_small"
    target_path= "/home/tobias/Dokumente/saved_context_vectors/paratest"
    files = get_paths_of_files(path, filetype=".txt")
    procs = 1
    k = 2
    dim = 500
    context_size = 2
    for i in range(0,len(files),procs):
=======
    base_path = "/home/tobias/Dokumente/testdata/wortschatz_small/"
    target_path= "/home/tobias/Dokumente/models/wortschatz/working"
    procs = 3
    k = 3
    dim = 1500
    context_size = 3
    num_folder = len(next(os.walk(base_path))[1])
    f_n = 0
    print(num_folder)
    # for source_path in next(os.walk(base_path))[1]:
    #
    #     print(source_path)
    #     print("total process\n")
    #     print("\r%f %%" % (100 * f_n / num_folder), end="")
    #     f_n +=1
    source_path = base_path# + source_path
    files = get_paths_of_files(base_path, filetype=".txt")
    if not os.path.exists(target_path):
        os.makedirs(target_path)

    size = len(files)
    print(size)

    for i in range(0,len(files),procs):
        if i % 1 == 0:
            print("\r%f %%" % (100 * i / size), end="")
>>>>>>> b3fec38... freitag
        jobs = []
        for j in range(0, procs):
            if i+j < len(files):
                process = multiprocessing.Process(target=bm_on_file,
                                                  args=(files[i+j],i+j,context_size, dim, k))
<<<<<<< HEAD
            jobs.append(process)
=======
                jobs.append(process)
>>>>>>> b3fec38... freitag

        for j in jobs:
            j.start()
        # Ensure all of the processes have finished
        for j in jobs:
            j.join()
<<<<<<< HEAD
    # at the end merge models
    target_file = target_path + "/accu.model"
    bwhole(path=target_path,
           target_file=target_file)

if __name__ == "__main__":
    #main()
    rim1 = RIModel.RIModel(dim=1, k=1)# dummy
    rim2 = RIModel.RIModel(dim=1, k=1)# dummy
    rim1.load_model_from_file("/home/tobias/Dokumente/saved_context_vectors/d1500accu_2.model")
    rim2.load_model_from_file("/home/tobias/Dokumente/saved_context_vectors/oanc/d1500k3written_1.pkl")
    print(rim1.dim,rim2.dim)
    merge_model(rim1,rim2)
    #            rim2.load_model_from_file("/home/tobias/Dokumente/saved_context_vectors/clob_crown/d1500k3merge.pkl"))
    rim1.write_model_to_file("/home/tobias/Dokumente/saved_context_vectors/d1500accu_2.model")
    # bwhole(path="/home/tobias/Dokumente/saved_context_vectors/paratest",
    #        target_file="/home/tobias/Dokumente/saved_context_vectors/paratest/accu")
=======
        # # at the end merge models
    target_file = os.path.split(target_path)[0] \
                  + "/merged"+ "/" + os.path.split(base_path)[1].replace('.','_')\
                  + ".model"
    # merge_models_in_folder(path=target_path,
    #                          target_file=target_file)
    # from shutil import rmtree
    # rmtree(target_path)


if __name__ == "__main__":
    #split_text_file("/home/tobias/Dokumente/testdata/wortschatz_small/2008/eng_news_2008_10K-sentences.txt",10)
    # start = time.time()
    # main()
    # end = time.time() - start
    # print("took me {} s".format(end))
    # merge_models_in_folder(path="/home/tobias/Dokumente/saved_context_vectors/wortschatz/merged",
    #                        target_file="/home/tobias/Dokumente/saved_context_vectors/wortschatz/dim1500k3c3.model")

    #
    # rim1 = RIModel.RIModel(dim=1, k=1)# dummy
    # rim2 = RIModel.RIModel(dim=1, k=1)# dummy
    # rim1.load_model_from_file("/home/tobias/Dokumente/models/state_of_union_1500k3c3.model")
    # rim2.load_model_from_file("/home/tobias/Dokumente/models/oanc_1500k3c3_merge.model")
    # print(rim1.dim,rim2.dim)
    # merge_model(rim1,rim2)
    # # #            rim2.load_model_from_file("/home/tobias/Dokumente/models/clob_crown/d1500k3merge.pkl"))
    # rim1.write_model_to_file("/home/tobias/Dokumente/models/complete_merge1500k3c3.model")
    #bwhole(path="/home/tobias/Dokumente/saved_context_vectors/paratest",
    #       target_file="/home/tobias/Dokumente/saved_context_vectors/paratest/accu")


>>>>>>> b3fec38... freitag

