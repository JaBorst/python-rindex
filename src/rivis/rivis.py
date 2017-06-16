
import sys
sys.path.append('../')

from  rindex import *
from tsne.tsne import *
import numpy as np
import pickle
from termcolor import colored
from matplotlib import colors



from bokeh.palettes import RdBu3, Spectral4

from bokeh.plotting import figure, output_file , show, ColumnDataSource
from bokeh.models import HoverTool, CategoricalColorMapper, ColorBar


class Rivis:

	rimodel_file = ""
	rimodel = RIModel.RIModel
	rimodel_matrix = []
	rimodel_keys = []


	tsne_file = ""
	tsne_model = np.array

	bokeh_infos = {}
	bokeh_allowed_infos = ["labels", "colors"]

	rivis_dump = "rivis.dump"


	X = []
	Y = []
	Z = []


	def __init__(self, rim = None):
		if not rim:
			self.rimodel = rim

	def set_model_file(self, filepath):
		self.rimodel_file = filepath
		if self.rimodel:
			self.rimodel.load_model_from_file(filepath)
		else:
			self.rimodel = RIModel.RIModel(dim=1,k=1)
			self.rimodel.load_model_from_file(filepath)


		self.rimodel_keys, self.rimodel_matrix = self.rimodel.to_matrix()


	def tsne2_calc(self):
		nm = np.array(self.rimodel_matrix)
		max = np.amax(np.absolute(nm))
		nm = nm / max


		tsne_matrix = tsne.tsne(X=nm, no_dims=2, perplexity=40.0)

		self.X = tsne_matrix[:, 0]
		self.Y = tsne_matrix[:, 1]

	def tsne2_load(self, modelfile =""):
		self.tsne_file = modelfile

		with open(modelfile, 'rb') as inputTSNE:
			self.tsne_matrix = pickle.load(inputTSNE)

		self.X = self.tsne_matrix[:, 0]
		self.Y = self.tsne_matrix[:, 1]


	def tsne2_scatter_plot(self):
		hover = HoverTool(
			tooltips=[
				("index", "$index"),
				("(x,y)", "($x, $y)"),
				("t", "@labels")
			])

		source = ColumnDataSource(
			data=dict(
				x=self.X,
				y=self.Y,
				labels=self.bokeh_infos.get("labels", []),
			)
		)

		if not self.bokeh_infos.get("colors", False):
			print("Automatic Colormapping")
			label_set = list(set(self.bokeh_infos.get("labels", [])))
			print(label_set)
			ccm = CategoricalColorMapper(factors=label_set, palette=[RdBu3[2], RdBu3[0]])

			p = figure(title="simple line example", x_axis_label='x', y_axis_label='y', tools=[hover])
			p.circle('x', 'y',
					 source=source,
					 fill_alpha=0.6,
					 fill_color={'field': 'labels', 'transform': ccm},
					 legend='labels',
					 line_color=None,
					 radius=0.5)
			p.legend.click_policy = "mute"
		show(p)

	def tsne2_layer_plot(self):

		hover = HoverTool(
			tooltips=[
				("index", "$index"),
				("(x,y)", "($x, $y)"),
				("t", "@labels")
			])

		source = ColumnDataSource(
			data=dict(
				x=self.X,
				y=self.Y,
				labels=self.bokeh_infos.get("labels", []),
			)
		)



		if not self.bokeh_infos.get("colors", False):
			print("Automatic Colormapping")
			label_set = list(set(self.bokeh_infos.get("labels", [])))
			print(label_set)
			n= np.array(self.bokeh_infos.get("labels", []))
			Xnp = np.array(self.X)
			Ynp = np.array(self.Y)
			dataX = {}
			dataY = {}
			for l in label_set:
				dataX[l] = Xnp[np.where(n == l)]
				dataY[l] = Ynp[np.where(n == l)]
			p = figure(title="simple line example", x_axis_label='x', y_axis_label='y', tools=[hover])

			print(label_set)
			print(RdBu3)
			for l, c in zip(label_set,  Spectral4):
				print(l)
				p.circle(dataX[l], dataY[l],  line_width=2, color=c, alpha=0.8,
					   muted_color=c, muted_alpha=0.2, legend=l)

			p.legend.location = "top_left"
			p.legend.click_policy = "mute"
			show(p)

			#
			#
			# ccm = CategoricalColorMapper(factors=label_set, palette=[RdBu3[2], RdBu3[0]])
			#
			# p.circle('x', 'y',
			# 		 source=source,
			# 		 fill_alpha=0.6,
			# 		 fill_color={'field': 'labels', 'transform': ccm},
			# 		 legend='labels',
			# 		 line_color=None,
			# 		 radius=0.5)
			# p.legend.click_policy = "mute"




	def tsne2_image_plot(self):
		topicNames = list(set(self.bokeh_infos.get("labels", [])))
		colorArray = []
		colorMap = {}

		numTopics = len(topicNames)
		print(topicNames)
		for i in range(0, len(topicNames)):
			colorMap[topicNames[i]] = list(colors.cnames.keys())[(i * 19) % 31]

		for e in self.bokeh_infos.get("labels", []):
			colorArray.append(colorMap[e])
		p = figure(title="simple line example", x_axis_label='x', y_axis_label='y')
		p.image(image=colorArray, x=np.min(self.X),
				y=np.min(self.Y),
				dw=np.max(self.X)-np.min(self.X),
				dh=np.max(self.Y)-np.min(self.Y),
				palette="Spectral11")
		show(p)


	def set_visualisation_dict(self, name = "", data = []):

		if name in self.bokeh_allowed_infos:
			self.bokeh_infos[name] = data
		else:
			print(colored("Don't know what %s is" % (name) , "red"))


	def load_labels(self, filepath):
		with open(filepath, 'rb') as inputFile:
			labels = pickle.load(inputFile)
			self.set_visualisation_dict("labels" , labels)




	def info(self):
		print("\n\n______________________________________________________________________________")
		print ("Info about Rivis Visualization Object")
		print ("Current Model: %s" % (self.rimodel_file) )
		print ("Current DataSize: X:%i  Y:%i  Z:%i" % (len(self.X), len(self.Y), len(self.Z)))
		print ("Current Number Keys: %i  \t%s" % (len(self.rimodel_keys), ", ".join(self.rimodel_keys[:5]) + ", ..." ))
		print("")
		if self.tsne_file != "":
			print("TSNE-Model set %s" %(self.tsne_file))
			try:
				if self.tsne_model.any():
					print("... and loaded")
			except:
				pass

		print("Bokeh Information for Visualisation")
		for i in self.bokeh_infos.keys():
			print("\t%s set...%i" %(i, len(self.bokeh_infos[i])))


		print("______________________________________________________________________________\n")

	def Load(self):
		f = open(self.rivis_dump,'rb')
		tmp_dict = pickle.load(f)
		f.close()

		self.__dict__.update(tmp_dict)


	def Save(self):
		f = open(self.rivis_dump,'wb')
		pickle.dump(self.__dict__,f,2)
		f.close()


if __name__ == "__main__":


	riv = Rivis()
	#riv.set_model_file("/home/jb/git/python-rindex/src/misc/reuters.model")
	riv.tsne2_load("/home/jb/git/python-rindex/src/misc/tmp/reuters.tsne")
	riv.load_labels("/home/jb/git/python-rindex/src/misc/reuters.topics")
	riv.tsne2_layer_plot()
	riv.info()