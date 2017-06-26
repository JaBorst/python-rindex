
import sys
sys.path.append('../')

from  rindex import *
from tsne.tsne import *
import numpy as np
import pickle
from termcolor import colored
from matplotlib import colors
from sklearn.neighbors import KDTree


from bokeh.palettes import RdBu3, Spectral4

from bokeh.plotting import figure, output_file , show, ColumnDataSource
from bokeh.models import HoverTool, CategoricalColorMapper, ColorBar


class Rivis:
	name = ""

	rimodel_file = ""
	rimodel = RIModel.RIModel
	rimodel_matrix = []
	rimodel_keys = []
	rimodel_kdtree = KDTree


	tsne_file = ""
	tsne_model = np.array

	bokeh_infos = {}
	bokeh_allowed_infos = ["labels", "colors"]

	bokeh_output_file_name = "rivis"
	bokeh_tools = "pan,wheel_zoom,box_zoom,reset"

	rivis_dump = "rivis.dump"




	X = []
	Y = []
	Z = []


	def __init__(self, rim = None):
		if rim:
			self.rimodel = rim
			self.rimodel_keys, self.rimodel_matrix = self.rimodel.to_matrix()

	def set_title(self, title = ""):
		'''
		Set the Title for all outputfiles and plot titles
		:param title:
		:return:
		'''
		self.name = title
		self.bokeh_output_file_name = title + ".html"
		self.tsne_file = title + ".tsne"
		self.rivis_dump = title + ".dump"
		self.rimodel_file = title + ".model"

	def set_model_file(self, filepath):
		'''
		You Can read in a RIModel from File
		:param filepath:
		:return:
		'''
		self.rimodel_file = filepath
		self.rimodel = RIModel.RIModel()
		self.rimodel.load_model_from_file(filename=filepath)

		self.rimodel_keys, self.rimodel_matrix = self.rimodel.to_matrix()

	def truncate(self, treshold = 0.1):
		self.rimodel.truncate(threshold=treshold)
		self.rimodel_keys, self.rimodel_matrix = self.rimodel.to_matrix()

	def tsne2_calc(self, iterations=1000):
		'''
		This calculates a tsne Reduction to 2 Dimensions and saves the plotable Data internally
		:param iterations: NUmber of Iterations for the tsne algorithm
		:return:
		'''
		if len(self.rimodel_matrix) == 0:
			if not self.rimodel:
				rimodel_keys, self.rimodel_matrix = self.rimodel.to_matrix()


		nm = np.array(self.rimodel_matrix)
		max = np.amax(np.absolute(nm))
		nm = nm / max

		tsne_matrix = tsne(X=nm, no_dims=2, perplexity=40.0, iter=iterations)

		self.X = tsne_matrix[:, 0]
		self.Y = tsne_matrix[:, 1]

	def pca2_calc(self):
		'''
		This calculates a PCA Reduction to 2 Dimensions and saves the plotable Data internally
		:param iterations: NUmber of Iterations for the tsne algorithm
		:return:
		'''
		if len(self.rimodel_matrix) == 0:
			if not self.rimodel:
				self.rimodel_keys, self.rimodel_matrix = self.rimodel.to_matrix()
		from sklearn.decomposition import PCA
		pca = PCA(n_components=2)
		red_data = pca.fit_transform(self.rimodel_matrix)
		self.is_sparse = False

		self.X = red_data[:, 0]
		self.Y = red_data[:, 1]



	def tsne2_load(self, modelfile =""):
		'''
		You can load already calculated tsne reductions by file
		:param modelfile:
		:return:
		'''
		self.tsne_file = modelfile

		with open(modelfile, 'rb') as inputTSNE:
			self.tsne_matrix = pickle.load(inputTSNE)

		self.X = self.tsne_matrix[:, 0]
		self.Y = self.tsne_matrix[:, 1]


	def scatter_plot(self, legend=True):
		'''
		This function creates a simple scatter plot of all the data
		:param legend: True/False
		:return:
		'''
		output_file(self.bokeh_output_file_name)
		hover = HoverTool(
			tooltips=[
				("index", "$index"),
				("t", "@labels")
			])

		source = ColumnDataSource(
			data=dict(
				x=self.X[:500],
				y=self.Y[:500],
				labels=self.bokeh_infos.get("labels", self.rimodel_keys)[:500],
			)
		)
		if not self.bokeh_infos.get("labels", False):
			print("No Lables set... Using Keys...\nNo Colormapping")


			p = figure(title=self.name, x_axis_label='x', y_axis_label='y', tools=self.bokeh_tools)
			p.add_tools(hover)
			p.circle('x', 'y',
					 source=source,
					 fill_alpha=0.8,
					 fill_color='black',
					 legend=False,
					 line_color=None,
					 radius=0.05)

		elif not self.bokeh_infos.get("colors", False):
			print("Automatic Colormapping")
			label_set = list(set(self.bokeh_infos.get("labels", [])))
			print(label_set)
			ccm = CategoricalColorMapper(factors=label_set, palette=[RdBu3[2], RdBu3[0]])

			p = figure(title="simple line example",
					   x_axis_label='x',
					   y_axis_label='y',
					   tools=self.bokeh_tools)
			p.add_tools(hover)
			p.circle('x', 'y',
					 source=source,
					 fill_alpha=0.6,
					 fill_color={'field': 'labels', 'transform': ccm},
					 legend=False,
					 line_color=None,
					 radius=0.5)
			p.legend.click_policy = "mute"


		show(p)

	def layer_plot(self):
		'''
		This creates a layer Plot of the data. Useful if labels or categorization available
		:return:
		'''
		output_file(self.bokeh_output_file_name)

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




	def image_plot(self):
		'''
		Not yet useful
		:return:
		'''
		output_file(self.bokeh_output_file_name)
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
		'''
		Set different visualisation infos like lables or color arrays
		:param name: name of the data ( must be in ["labels", "colors"] )
		:param data:
		:return:
		'''
		if name in self.bokeh_allowed_infos:
			self.bokeh_infos[name] = data
		else:
			print(colored("Don't know what %s is" % (name) , "red"))


	def load_labels(self, filepath):
		'''
		load labels from file
		:param filepath:
		:return:
		'''
		with open(filepath, 'rb') as inputFile:
			labels = pickle.load(inputFile)
			self.set_visualisation_dict("labels" , labels)



	def info(self):
		'''
		Prints Information about the stored data object
		:return:
		'''
		print("\n\n______________________________________________________________________________")
		print ("Info about Rivis Visualization Object")
		print ("Rivis Dump Name : %s" % (self.rivis_dump) )
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

	def build_tree(self):
		self.rimodel.reduce_dimensions(method="truncated_svd", target_size=50)
		keys, self.rimodel_kdtree = self.rimodel.to_tree(method="minkowski", leaf_size=50)


	def plot_similar(self, key="", number=-1, similarity_measure="jsd", circle_size=0.01):
		#if self.rimodel_kdtree != None:
		#	pass
		#else:
		if number < 0:
			number = len(self.rimodel_keys)
		plotKeys = self.rimodel.is_similar_to(word=key, method=similarity_measure, count=number, silent=True)

		X_sim = [self.X[self.rimodel_keys.index(key)]]
		Y_sim = [self.Y[self.rimodel_keys.index(key)]]
		plotKeyList = [key] + [y[1] for y in plotKeys]
		similarities = [1]
		colors = ["red"]
		for k in plotKeys:
			X_sim.append(self.X[self.rimodel_keys.index(k[1])])
			Y_sim.append(self.Y[self.rimodel_keys.index(k[1])])
			colors.append( "rgb(%i,%i,%i)" % (int(255-k[0]*255), int(255-k[0]*255), int(255-k[0]*255)))
			similarities.append(k[0])

		print(colors)
		output_file(self.bokeh_output_file_name)
		hover = HoverTool(
			tooltips=[
				("index", "$index"),
				("x", "@x"),
				("y", "@y"),
				("similarity","@sim")
			])

		source = ColumnDataSource(
			data=dict(
				x=plotKeys,
				y=plotKeys,
				labels=self.bokeh_infos.get("labels", []),
				sim=similarities,
			)
		)

		source = ColumnDataSource(
			data=dict(
				x=X_sim,
				y=Y_sim,
				labels=plotKeyList,
				sim = similarities,

			)
		)
		if not self.bokeh_infos.get("labels", False):
			print("No Lables set... Using Keys...\nNo Colormapping")

			p = figure(title=self.name, x_axis_label='x', y_axis_label='y', tools=self.bokeh_tools)
			p.add_tools(hover)
			p.circle('x', 'y',
					 source=source,
					 fill_alpha=0.8,
					 legend=False,
					 fill_color=colors,
					 line_color=None,
					 radius=circle_size)


		show(p)

	def plot_similarity_matrix(self, ):
		similarities = self.rimodel.most_similar(count=-1, silent=True, method="jsd")
		plotKeys = list(set([ p[0] for (k,p) in similarities] + [ p[1] for (k,p) in similarities]))
		plotKeys.sort()
		print("Length: ", len(plotKeys))
		#turn sim into dictionary for fast lookup
		simDic = {}

		for (sim, pair) in similarities:
			simDic[pair] = sim

		d = []
		xdata = []
		ydata = []
		plot_similarities =[]
		for k1 in plotKeys:
			color_row = []
			label_row = []
			for k2 in plotKeys:
				s=0
				if k1 == k2 :
					s = 1
				else:
					s = simDic.get((k1,k2),0)
					if s == 0:
						s = simDic.get((k2,k1),0)

				#color_row.append(s)
				d.append( "rgb(%i, %i, %i)" % (int(255-s*255),int(255-s*255),int(255-s*255)))
				xdata.append(k1)
				ydata.append(k2)
				plot_similarities.append(s)

		hover = HoverTool(
			tooltips=[
				("index", "$index"),
				("x", "@x"),
				("y", "@y"),
				("similarity", "@sim"),

			])

		source = ColumnDataSource(
			data=dict(
				x=xdata,
				y=ydata,
				sim=plot_similarities,
			)
		)
		p = figure(x_range=plotKeys, y_range=plotKeys)

		# must give a vector of image data for image parameter
		#p.rect(x=xdata, y=ydata, fill_color=d, width=1, height=100, line_color=None, height_units="screen")
		p.square('x', 'y', source=source, fill_color=d, size=25, line_color=None)
		p.add_tools(hover)
		output_file("image.html", title="image.py example")

		show(p)  # open a browser
	def image_similarity_matrix(self, ):
		similarities = self.rimodel.most_similar(count=-1, silent=True, method="jsd")
		plotKeys = list(set([ p[0] for (k,p) in similarities] + [ p[1] for (k,p) in similarities]))
		plotKeys.sort()
		print("Length: ", len(plotKeys))
		#turn sim into dictionary for fast lookup
		simDic = {}

		for (sim, pair) in similarities:
			simDic[pair] = sim

		d = []
		xdata = []
		ydata = []
		plot_similarities =[]
		img = np.zeros((len(plotKeys), len(plotKeys), 3), dtype=np.uint8)

		for i, k1 in enumerate(plotKeys):
			color_row = []
			label_row = []
			for j, k2 in enumerate(plotKeys):
				s=0
				if k1 == k2 :
					s = 1
				else:
					s = simDic.get((k1,k2),0)
					if s == 0:
						s = simDic.get((k2,k1),0)

				img[i][j][0] = int(255-s*255)
				img[i][j][1] = int(255-s*255)
				img[i][j][2] = int(255-s*255)

		from scipy.misc import imshow, imsave
		imshow(img)
		imsave("image.png", img)

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
	riv.set_model_file("/home/jb/git/python-rindex/src/models/letterview.model")
	#riv.load_labels("/home/jb/git/python-rindex/src/misc/reuters.topics")
	#riv.set_visualisation_dict("label2", riv.bokeh_infos["label"])
	#riv.tsne2_load("/home/jb/git/python-rindex/src/misc/tmp/reuters.tsne")
	#riv.load_labels("/home/jb/git/python-rindex/src/misc/reuters.topics")
	#riv.tsne2_layer_plot()
	#riv.info()
	riv.image_similarity_matrix()
