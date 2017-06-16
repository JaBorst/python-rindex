
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

from bokeh.plotting import figure, output_file , show, ColumnDataSource
from bokeh.models import HoverTool

tmpdir="tmp/"

def tsneOnModel(name="dump.model"):

	print("Loading Model File")
	r = RIModel.RIModel(100,2)
	try:
		r.load_model_from_file(tmpdir + name + ".model")
	except:
		print("File not Found: ", tmpdir + name + ".model")
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
	with open(tmpdir + name + ".tsne", "wb") as outputTsne:
		pickle.dump(Y,outputTsne)
		print("TSNE successfully dumped!")


	plotOnly(name + ".tsne")



def plotOnly(name):

	print("Loading TSNE File")
	Y = np.array
	with open(tmpdir + name+".tsne", 'rb') as inputTSNE:
		Y = pickle.load(inputTSNE)


	labels = []
	with open(tmpdir + name+".topics", 'rb') as inputFile:
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
	output_file("tmp/index.html")

	hover = HoverTool(
		tooltips=[
		("index", "$index"),
		("(x,y)", "($x, $y)"),
		("t", "@desc")
	])

	source = ColumnDataSource(
		data=dict(
			x=Y[:,0],
			y=Y[:,1],
			desc=labels,
		)
	)

	# create a new plot with a title and axis labels
	p = figure(title="simple line example", x_axis_label='x', y_axis_label='y', tools=[hover])

	# add a line renderer with legend and line thickness
	p.circle('x', 'y', legend='labels',source=source,fill_color= colorArray, fill_alpha=0.6, line_color=None, radius=1)

	# show the results
	show(p)




def main():
	print(sys.argv)

	if len(sys.argv) == 3:
		if sys.argv[1] == 'c' or sys.argv[1] == "calc":
			tsneOnModel(sys.argv[2])
		elif sys.argv[1] == 'p' or sys.argv[1] == "plot":
			plotOnly(sys.argv[2])
	else:
		print("Usage: python visualize.py [c(alc)|p(lot)] [name] ")
		print("python visualize.py calc [name] will look for a [name].model and a [name].topics file and calculate a tsne, dump a [name].tsne and plot it")
		print("python visualize.py plot [name] will loog for [name].tsne and [name].topics and just plot")


if __name__ == "__main__":
	main()