#
# __filename__: build_model.py
#
# __description__: The functions in this file call methods in RIModel.py
# to build random-index models of given (text-)files. There are two kinds
# of models: The first kind, called by analyze_file_by_context() or
# analyze_text_files_of_folder() creates context-vectors for each word.
# Document-Models, called by build_doc_vec() and build_doc_model()
# add up the context-vectors of one document to get a representation
# of the same.
#
# __remark__: Mind the different methods to add context vectors!
#
# Created by Tobias Wenzel on 08-08-17.
# Copyright (c) 2017 Tobias Wenzel

import sys
sys.path.append('../')
from nltk.tokenize import word_tokenize as w_tokenize
from nltk.tokenize import sent_tokenize as s_tokenize
from os.path import basename
import numpy as np

## for pdfs
#import PyPDF2

from t_helper import log_model
from t_helper import clean_word_seq
# to build sample model with random sentences of document
from t_helper import get_n_sents
from t_helper import get_paths_of_files

from rindex import *

# used for nice printings
line_length= 60
dotted_line = "-"*line_length
line = "_"*line_length
colon_line = ":"*line_length
quarter_space_line = " "*int(line_length/4)

"""
            HELPER-FUNCTIONS
"""


def vec_to_model(path="", target_file="", dim=1500):
        """
        if vecs for documents were created as seperate
        .vec files.
        :param path:
        :param target_file:
        :return:
        """
        model_files = get_paths_of_files(path, filetype=".vec")
        rim1 = RIModel.RIModel(dim=dim, k=3)  # dummy

        print("start merging.")

        size = len(model_files)
        for i in range(1, len(model_files)):
            if i % 10 == 0:
                print("\r%f %%" % (100 * i / size), end="")
            if model_files[i] == target_file:
                continue
            with open(model_files[i] , 'rb') as inputFile:
                rim1.ContextVectors[basename(model_files[i])[:-4]] = pickle.load(inputFile)

        rim1.write_model_to_file(target_file)

def get_weight_vector(context_size=2, target_index=1):
    """
    e.g. 0.75 0 0.75 0.25
    :param context_size:
    :param target_index:
    :return:
    """
    weights=np.zeros(context_size)
    step_size = 1 / context_size
    weights[:target_index] = np.linspace(1 - step_size, 1, num=target_index)
    weights[target_index + 1:] = np.linspace(1, step_size, num=context_size-target_index - 1)
    return  weights


"""
            BUILD-MODEL-FUNCTIONS
            1) Word-Similarity
"""


def analyze_file_by_context(filename, rmi, context_size=2):
    """

    :param filename:
    :param rmi:
    :param context_size:
    :return:
    """
    # print("calls add_context!")
    index = 0
    weights = get_weight_vector(context_size, target_index=index)
    try:
        with open(filename, 'r', encoding='utf-8') as fin:
            text = fin.read()
    except:
        try:
            with open(filename, 'r', encoding='iso-8859-1') as fin:
                text = fin.read()
        except:
            print("error with ", filename)
            return
    context = s_tokenize(text)
    size = len(context)
    for i, sentence in zip(range(len(context)), context):
        if i % 100 == 0:
            print("\r%f %%" % (100 * i / size), end="")
        sent = clean_word_seq(w_tokenize(sentence))
        try:
            for j in range(len(sent)):
                context = []
                for k in range(j, j + context_size):
                    try:
                        context.append(sent[k])
                    except:
                        pass
                if len(context):
                    try:
                        # first creates iv by contex, second incr.
                        # rmi.add_context(context, index=index)
                        rmi.add_unit(context[index], context, weights=weights)
                    except:
                        pass
        except:
            pass
            # print("finished analyze_file_by_context")

@log_model
def analyze_text_files_of_folder(rmi, source_path="", contextSize=2, ext=""):
    """
    walks through folder and builds a model
    :param:rmi
    :param path:
    :param contextSize:
    :param ext:
    :return:
    """
    files = get_paths_of_files(source_path, filetype=".txt")
    if len(files) == 0:
        print("not txt-files in folder")
        return
    size = len(files)
    for i, filename in zip(range(len(files)), files):
        if i % 10 == 0:
            print("\r%f %%" % (100 * i / size), end="")
        analyze_file_by_context(filename, rmi, context_size=contextSize)
        if i % 50 == 0:
            rmi.write_model_to_file(ext)
    rmi.write_model_to_file(ext)
    print("finished analyze_text_files_of_folder-model")


"""
            BUILD-MODEL-FUNCTIONS
            2) Document-Similarity
"""


def build_doc_vec(filename="", dim=1500, k=3, context_size=3, index=1):
    """

    :param filename:
    :param dim:
    :param k:
    :param context_size:
    :param index:
    :return:
    """
    weights = get_weight_vector(context_size, target_index=index)
    print("with weight-vector:", weights)

    with open(filename, 'r', encoding='utf-8') as fin:
        full_text = fin.read()
    try:
        base_filename = basename(filename)  # [:-4] ohne file-endung
        print(base_filename)
        rmi = RIModel.RIModel(dim, k)

        doc_text = s_tokenize(full_text)
        # doc_text = get_n_sents(500)
        num_sents = len(doc_text)

        for z, sentence in zip(range(num_sents), doc_text):
            if z % 100 == 0:
                print("\r%f %%" % (100 * z / num_sents), end="")
            sent = clean_word_seq(w_tokenize(sentence))
            try:
                for j in range(len(sent)):
                    context = []
                    for w in range(j, j + context_size):
                        try:
                            context.append(sent[w])
                        except:
                            pass
                    if len(context):
                        try:
                            rmi.add_context(context, index=index)  # for actual context
                            rmi.add_unit(context[index], context, weights=weights)  # for word context
                        except:
                            pass
            except Exception as e:
                pass
        compr = sp.coo_matrix((rmi.dim, 1))
        for key in rmi.ContextVectors.keys():
            compr += rmi.ContextVectors[key]

        return compr
    except Exception as e:
        print("error", filename, e)


@log_model
def build_doc_model(document_rmi, source_path="" \
                    , context_size=3, dim=1500, k=10, index=1):
    files = get_paths_of_files(source_path, filetype="")
    size = len(files)
    for i, filename in zip(range(size), files):
        document_rmi.ContextVectors[basename(filename)] = build_doc_vec(filename, dim=dim, k=k,
                                                                        context_size=context_size, index=index)


def main():
    dim = 1500
    k = 3
    rmi = RIModel.RIModel(dim, k)
    context_size = 3
    rmi.is_sparse = True
    print(colon_line)
    print(quarter_space_line, "Build Random Indexing Model")
    print(colon_line)
    mode = input("\t[0]: no \t 1: yes \t 2: update doc-model\n") or 0

    if int(mode) == 1:

        file_source = "/home/tobias/Downloads/Martin_Luther_Uebersetzung_1912.txt"
        folder_source = "/home/tobias/Dokumente/testdata/Newspapers/Crown_RAW"
        doc_model_source_path= "/home/tobias/Dokumente/testdata/doc_modell"
        """
            call: build doc model
        """
        # build_doc_model(source_path=doc_model_source_path \
        #                            ,context_size=3, k=rmi.k, dim=rmi.dim, index=0\
        #                 , document_rmi=rmi)
        """
            call: build word model
        """
        # model_name = "/home/tobias/Dokumente/models/written2_1500k3c3.model"
        # analyze_text_files_of_folder(rmi, path=folder_source,
        #                              ext=model_name,
        #                              contextSize=context_size)
        #analyze_file_by_context(filename=file_source, rmi=rmi, context_size=context_size)
        # needs to save after!
    elif int(mode) == 2:
        """
            Update Doc Model
        """
        rmi_filename = "/home/tobias/Dokumente/models/doc-models/books/testing_books.model"
        rmi.load_model_from_file(rmi_filename)
        filename = "/home/tobias/Dokumente/testdata/doc_modell/shakespeare_romeo_and_juliet"
        rmi.ContextVectors[basename(filename)] = build_doc_vec(filename,
                                                               dim=rmi.dim, k=rmi.k,
                                                               context_size=context_size,
                                                               index=0)
        rmi.write_model_to_file(rmi_filename)




if __name__ == '__main__':
    main()