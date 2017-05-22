import sys
sys.path.append('../')
import multiprocessing

import pickle
from t_helper import get_paths_of_files
from t_helper import clean_word_seq
from nltk.tokenize import word_tokenize as w_tokenize
from nltk.tokenize import sent_tokenize as s_tokenize
from rindex import *
import os.path


def bm_on_file(filename, f_number, context_size=2, dim=1500, k=3):
    """
    
    :param filename: 
    :param f_number: 
    :param context_size: 
    :param dim: 
    :param k: 
    :return: 
    """
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
    sents = s_tokenize(text)
    size = len(sents)
    rmi = RIModel.RIModel(dim, k)
    # rmi.is_sparse = True
    for i, sent in zip(range(len(sents)), sents):
    #     if i % 100 == 0:
    #         print("\r%f %%" % (100 * i / size), end="")
        sent = clean_word_seq(w_tokenize(sent))
        if not sent:
            continue
        for j in range(len(sent)):
            context = []
            for k in range(j, j + context_size):
                try:
                    context.append(sent[k])
                except:
                    pass
            if len(context):
                try:
                    rmi.add_context(context, index=0)
                except:
                    pass
                    # break
    fout = "/home/tobias/Dokumente/saved_context_vectors/newsgroups/working/word_sim"+str(f_number)+".model"
    rmi.write_model_to_file(fout)
    # break


# def merge_model(self, rim2):
#     """
#     :param path
#     :return:
#     """
#     if self.dim != rim2.dim or self.k != rim2.k:
#         print("dim,k not matching")
#         return
#
#     for key in self.ContextVectors.keys():
#         if key in rim2.ContextVectors.keys():
#             self.ContextVectors[key] = self.ContextVectors[key] + rim2.ContextVectors[key]
#         else:
#             self.ContextVectors[key] = self.ContextVectors[key]
#     for key in rim2.ContextVectors.keys():
#         if key not in self.ContextVectors.keys():
#             self.ContextVectors[key] = rim2.ContextVectors[key]
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



def merge_models_in_folder(path="", target_file=""):
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
    size= len(model_files)
    for i in range(1,len(model_files)):
        if i % 10 == 0:
            print("\r%f %%" % (100 * i / size), end="")
        if model_files[i] == target_file:
            continue
        rim2.load_model_from_file(model_files[i])
        merge_model(rim1=rim1,rim2=rim2)

    rim1.write_model_to_file(target_file)
    # rim1.reduce_dimensions(method="truncated_svd", target_size=50)
    # rim1.write_model_to_file(target_file)
    #
    # keys, kdt = rim1.to_tree(method="minkowski", leaf_size=50)
    # with open("/home/tobias/Dokumente/saved_context_vectors/newsgroups/word_sim.tree", 'wb') as output:
    #     pickle.dump(kdt, output)
    # with open("/home/tobias/Dokumente/saved_context_vectors/newsgroups/word_sim.keys", 'wb') as output:
    #     pickle.dump(keys, output)


def main():
    base_path = "/home/tobias/Dokumente/testdata/20_newsgroups/"
    target_path= "/home/tobias/Dokumente/saved_context_vectors/newsgroups/working"
    procs = 1
    k = 3
    dim = 1500
    context_size = 2
    num_folder = len(next(os.walk(base_path))[1])
    f_n = 0
    for source_path in next(os.walk(base_path))[1]:
        print(source_path)
        print("total process\n")
        print("\r%f %%" % (100 * f_n / num_folder), end="")
        f_n +=1
        source_path = base_path + source_path
        files = get_paths_of_files(source_path, filetype="")
        if not os.path.exists(target_path):
            os.makedirs(target_path)

        size = len(files)

        for i in range(0,len(files),procs):
            if i % 10 == 0:
                print("\r%f %%" % (100 * i / size), end="")
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
        # # at the end merge models
        target_file = os.path.split(target_path)[0] \
                      + "/merged"+ "/" + os.path.split(source_path)[1].replace('.','_')\
                      + ".model"
        merge_models_in_folder(path=target_path,
                               target_file=target_file)
        from shutil import rmtree
        rmtree(target_path)


if __name__ == "__main__":
    # main()
    # merge_models_in_folder(path="/home/tobias/Dokumente/saved_context_vectors/newsgroups/merged",
    #                        target_file="/home/tobias/Dokumente/saved_context_vectors/newsgroups/merge.model")

    rim1 = RIModel.RIModel(dim=1, k=1)# dummy
    rim2 = RIModel.RIModel(dim=1, k=1)# dummy
    rim1.load_model_from_file("/home/tobias/Dokumente/saved_context_vectors/d1500accu_no_stemming.model")
    rim2.load_model_from_file("/home/tobias/Dokumente/saved_context_vectors/newsgroups/merge.model")
    print(rim1.dim,rim2.dim)
    merge_model(rim1,rim2)
    #            rim2.load_model_from_file("/home/tobias/Dokumente/saved_context_vectors/clob_crown/d1500k3merge.pkl"))
    rim1.write_model_to_file("/home/tobias/Dokumente/saved_context_vectors/d1500accu_no_stemming.model")
    #bwhole(path="/home/tobias/Dokumente/saved_context_vectors/paratest",
    #       target_file="/home/tobias/Dokumente/saved_context_vectors/paratest/accu")



