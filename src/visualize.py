from  rindex import *
from tsne.tsne import *
import numpy as np
import pylab as Plot
import sys



def tsneOnModel(path="dump.model"):

	print("Loading Model File")
	r = RIModel.RIModel(100,2)
	try:
		r.loadModelFromFile(path)
	except:
		print("File not Found")
		exit()



	dim = r.ContextVectors[list(r.ContextVectors.keys())[0]].shape[0]


	print("Dimension: ", dim)

	print("Converting Model to Math array")
	m = []
	for v in r.ContextVectors.values():
		m.append(v.transpose().toarray()[0])
	nm = np.array(m)

	Y = tsne(nm, 2, dim, 20.0);
	Plot.scatter(Y[:, 0], Y[:, 1], 20, [1,2]);
	Plot.show();




def main():
	print(sys.argv)

	if len(sys.argv) == 2:
		tsneOnModel(sys.argv[1])
	elif len(sys.argv) == 1:
		tsneOnModel()
	else:
		print("Arguments?")


if __name__ == "__main__":
	main()