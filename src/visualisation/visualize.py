
import sys
sys.path.append('../')

from  rindex import *
from tsne.tsne import *
import numpy as np
import pylab as Plot
import sys
import pickle
from sklearn.preprocessing import normalize

import matplotlib.patches as mpatches

from matplotlib import colors
import pylab

tmpdir="tmp/"

def tsneOnModel(path="dump.model"):

	print("Loading Model File")
	r = RIModel2.RIModel(100,2)
	r.load_model_from_file(path)
	try:
		r.load_model_from_file(path)
	except:
		print("File not Found")
		exit()


	dim = r.dim


	print("Dimension: ", dim)
	print("Converting Model to Math array")
	m = []
	for v in r.ContextVectors.values():
		#print(v)
		m.append(v.transpose().toarray()[0])

	nm = np.array(m)
	max = np.amax(np.absolute(nm))
	print(max)
	nm = nm / max

	print(nm)
	Y = tsne(X=nm, no_dims=2, perplexity=40.0);
	with open(tmpdir + "reuters.tsne", "wb") as outputTsne:
		pickle.dump(Y,outputTsne)
		print("TSNE successfully dumped!")


	plotOnly(tmpdir + "reuters.tsne")



def plotOnly(path):

	print("Loading TSNE File")
	Y = np.array
	with open(path,'rb') as inputTSNE:
		Y = pickle.load(inputTSNE)


	labels = []
	with open(tmpdir + "reuters.topics", 'rb') as inputFile:
		labels = pickle.load(inputFile)

	topicNames= list(set(labels))
	colorArray = []
	colorMap = {}


	numTopics = len(topicNames)
	print(topicNames)
	print("Different Topics: ", numTopics)

	recs = []
	for i in range(0, len(topicNames)):
		colorMap[topicNames[i]] = list(colors.cnames.keys())[(i*19)%31]
		recs.append(mpatches.Rectangle((0, 0), 1, 1, facecolor=colors.cnames[colorMap[topicNames[i]]]))

	print("Colors done")

	for e in labels:
		colorArray.append(colorMap[e])

	print(type(Y))
	Plot.scatter(x=Y[:, 0], y=Y[:, 1],  c=colorArray)
	Plot.legend(recs, topicNames, loc=4)
	Plot.show()





def main():
	print(sys.argv)

	if len(sys.argv) == 3:
		if sys.argv[1] == 'c' or sys.argv[1] == "calc":
			tsneOnModel(sys.argv[2])
		elif sys.argv[1] == 'p' or sys.argv[1] == "plot":
			plotOnly(sys.argv[2])
	else:
		print("Arguments?")


if __name__ == "__main__":
	main()