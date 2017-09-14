import sys
sys.path.append('../')
#from rindex import *
import os
from nltk.corpus import stopwords
stop = set(stopwords.words('english'))
<<<<<<< HEAD

=======
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize as w_tokenize
from nltk.tokenize import sent_tokenize as s_tokenize
import numpy as np
>>>>>>> b3fec38... freitag

def clean_word_seq(context = []):
    """
    # some trimming
    :param context: 
    :return: 
    """
    if len(context) > 1:
        return [word.lower() for word in context if word.isalpha() and len(word) > 1 and word not in stop]
    elif len(context) == 0:
        return
    elif len(context) == 1:
        if context[0].isalpha() and context[0] and context[0] not in stop:
            return context[0].lower()  #stemmer.stem()
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
#
# def merge_dicts(d1={}, d2={}):
#     """
#     as the name suggests: if we have more than one dict/model
#     this is an incremental function to merge them
#     :param d1:
#     :param d2:
#     :return:
#     """
#     nn = {}
#     for key in d1.keys():
#         if key in d2.keys():
#             nn[key] = d1[key]+d2[key]
#         else:
#             nn[key] = d1[key]
#     for key in d2.keys():
#         if key not in d1.keys():
#             nn[key] = d2[key]
#     rim = RIModel(dim=1500,k=3)
#
#     rim.ContextVectors = nn
#     rim.write_model_to_file("accu")
#     print("finished merging")



# def normalize_matrix(matrix):
#     from sklearn.preprocessing import normalize
#     return normalize(matrix, axis=1, norm='l2',copy=False)


# def search_in_matrix(matrix, keys=[], word=""):
#     """
#
#     :param matrix:
#     :param keys:
#     :param word:
#     :return:
#     """
#     from scipy import spatial
#     word_iv = matrix[keys.index(word)]
#
#     if sp.issparse(word_iv):
#         word_iv = word_iv.toarray()
#
#     max_d = 0
#     max_key = ""
#     for key in keys:
#         if key != word:
#             d = 1 - spatial.distance.cosine(word_iv, matrix[keys.index(key)])  # .toarray())
#             if d > max_d:
#                 max_d = d
#                 max_key = key
#     print(max_d, max_key)


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

def get_authors(path, output_file):
    """
    only used to get authors + books
    :param output_file:
    :return:
    """

    import csv
    with open(output_file, 'w', newline='\n') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for book in get_paths_of_files(path=path,filetype=""):
            book_authour = book.split("/")[-1].split("_")
            #print(book_authour[0])
            writer.writerow([book_authour[0], " ".join(book_authour[1:])])



def write_dicct_to_csv(dicct={}, output_file=""):
    import csv
    with open(output_file, 'w', newline='\n') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        same_author_vals = []
        thres = 0.80
        for obj in dicct:
            first_author = obj[1][0].split("_")[0]
            second_author = obj[1][1].split("_")[0]
            if first_author != second_author:
                pass
            else:
                same_author_vals.append(obj[0])
                hit = 0
                if obj[0] >= thres:
                    hit = 1
                writer.writerow([obj[0],
                                 first_author, " ".join(obj[1][0].split("_")[1:]),
                                 second_author, " ".join(obj[1][1].split("_")[1:]),hit])



        different_author_vals = []
        writer.writerow([])  # values after should be small
        for obj in dicct:
            first_author = obj[1][0].split("_")[0]
            second_author = obj[1][1].split("_")[0]
            if first_author == second_author:
                pass
            else:
                hit = 0
                if obj[0] >= thres:
                    hit = 1
                different_author_vals.append(obj[0])
                writer.writerow([obj[0],
                                 first_author, " ".join(obj[1][0].split("_")[1:]),
                                 second_author, " ".join(obj[1][1].split("_")[1:]),hit])

        same_author_median = np.median(same_author_vals)
        diff_authour_median = np.median(different_author_vals)


        dists_1 = []
        sum_1 = 0
        for val in same_author_vals:
            if val > diff_authour_median:
                sum_1 += 1
                dists_1.append(np.abs(different_author_vals - val))
        sum_2 = 0
        dists_2 = []
        for val in different_author_vals:
            if val < same_author_median:
                sum_2 += 1
                dists_2.append(np.abs(same_author_median-val))


        print(colon_line)
        print("\tSame Author")
        print("correct range:\t",round(min(same_author_vals),3),round(max(same_author_vals),3))
        print("median:\t",same_author_median)
        print("sd",np.std(same_author_vals))
        print("distance to other median",np.median(dists_1))
        print("same_author_vals above diff_author_median",\
              100*sum_1/len(same_author_vals))
        print("{}/{}".format(sum_1,len(same_author_vals)))
        print(colon_line)

        print("\tDifferent Author")
        print("false range:\t", round(min(different_author_vals), 3), round(max(different_author_vals), 3))
        print("median",diff_authour_median)
        print("sd",np.std(different_author_vals))
        print("distance to other median",np.median(dists_2))
        print("diff_author_vals below same_author_median",\
              100*sum_2/len(different_author_vals))
        print("{}/{}".format(sum_2,len(different_author_vals)))


"""
    Testing
"""
def sentence_mean_sd(filename=""):
    """

    :param filename:
    :return:
    """
    filename = "/home/tobias/Dokumente/testdata/stateofunion.txt"

<<<<<<< HEAD
    rim.ContextVectors = nn
    rim.write_model_to_file("accu")
    print("finished merging")
=======
    with open(filename, 'r') as fin:
        text = fin.read()
    sents = s_tokenize(text)
    num_words_of_sents = np.zeros(len(sents))
    for i in range(len(sents)):
        num_words_of_sents[i] = len(w_tokenize(sents[i]))
    print(num_words_of_sents)
    return np.mean(num_words_of_sents), np.std(num_words_of_sents)

def split_state_of_union():
    """

    :return:
    """
    import re

    exp = "\*\*\*"
    filename = "/home/tobias/Dokumente/testdata/stateofunion.txt"

    with open(filename, 'r') as fin:
        text = []
        target_file = ""
        collect = False
        president = ""
        for line in fin:
            if re.match(exp, line):
                if text and president:
                    with open(target_file, 'a') as fout:
                        fout.writelines(text)
                text = []
                fin.readline()
                fin.readline()
                president = fin.readline().strip()
                text.append(president)
                target_file = "/home/tobias/Dokumente/testdata/presidents/" \
                              + president + ".txt"

            else:
                text.append(line)


def get_n_sents(n=100, filename=""):
    """
    to get text which is +- equal in size (or at least number of sents)
    :param n:
    :param filename:
    :return:
    """
    text = ""
    with open(filename, 'r') as fin:
        text = fin.read()
    sents = s_tokenize(text)
    indices = np.random.permutation(range(len(sents)))
    if len(sents) >= n:
        return [sents[indices[i]] for i in range(n)]
    elif 0 < len(sents):
        print("short file:", len(sents))
        return [sents[indices[i]] for i in range(len(sents))]
    else:
        print("no sentences found (?)")

# @depr
# def search_in_matrix(matrix, keys=[], word=""):
#     """
#     ...shouldn't be necessary by now
#     :param matrix:
#     :param keys:
#     :param word:
#     :return:
#     """
#     from scipy import spatial
#     word_iv = matrix[keys.index(word)]
#
#     if sp.issparse(word_iv):
#         word_iv = word_iv.toarray()
#
#     max_d = 0
#     max_key = ""
#     for key in keys:
#         if key != word:
#             d = 1 - spatial.distance.cosine(word_iv, matrix[keys.index(key)])  # .toarray())
#             if d > max_d:
#                 max_d = d
#                 max_key = key
#     print(max_d, max_key)

def create_test_case(filename):
    """
    creates files with decreasing number of sentences
    :param filename:
    :return:
    """
    with open(filename, 'r', encoding='utf-8') as fin:
        full_text = fin.read()
        doc_text = s_tokenize(full_text)

        original_length = len(doc_text)
        for i in range(original_length,0,-50):
            print(filename+str(i))
            with open(filename+str(i), 'w', encoding='utf-8') as fout:
                for j in range(0,i):
                    fout.write(doc_text[j])
                    fout.write(" ")

if __name__ == '__main__':
    #create_test_case("/home/tobias/Dokumente/testdata/plato-test/plato_apology")
    #print(sentence_mean_sd())
    get_authors(path="/home/tobias/Dokumente/testdata/doc_modell",output_file="/home/tobias/Dokumente/testdata/overview.csv")