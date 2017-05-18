import sys
sys.path.append('../')
from rindex import *
import os
from nltk.corpus import stopwords
stop = set(stopwords.words('english'))


def clean_word_seq(context = []):
    """
    # some trimming
    :param context: 
    :return: 
    """
    if len(context) > 1:
        return [word.lower() for word in context if word.lower() not in stop and word.isalpha()]
    elif len(context) == 0:
        return
    elif len(context) == 1:
        if context[0].lower() not in stop and context[0].isalpha():
            return context[0].lower()
        else:
            return


def get_paths_of_files(path = "", filetype=".txt"):
    """

    :param path: 
    :return: 
    """
    list_of_files = []
    i = 0
    for (dirpath, dirnames, filenames) in os.walk(path):
        for filename in filenames:
            if filename.endswith(filetype):
                list_of_files.append(os.sep.join([dirpath, filename]))
    return list_of_files

def merge_dicts(d1={}, d2={}):
    """
    as the name suggests: if we have more than one dict/model
    this is an incremental function to merge them
    :param d1:
    :param d2:
    :return:
    """
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
