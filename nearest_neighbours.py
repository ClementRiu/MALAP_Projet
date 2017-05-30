import numpy as np
import numpy.linalg as npl
import matplotlib.collections as mc
import matplotlib.pyplot as plt

import tools

class Nearest_Neighbours:
	def __init__(self, k=1, alpha=1):
		self.negativ = False
		self.k = k
		self.alpha = alpha

	def fit(self, data, label):
		data = data.reshape(len(label), -1)
		nb_ex, self.dim = data.shape
		ylab = set(label.flat)
		if len(ylab) > 2:
			print("pas bon nombres de labels (%d)" % (ylab,))
			return
		elif len(ylab) == 2:
			self.negativ = True
		self.label = label
		self.data = data

	def predict(self, data):
		self.lines1 = []
		self.lines2 = []

		data = data.reshape(-1, self.dim)
		diff = np.zeros((data.shape[0], self.data.shape[0]))		
		label = np.zeros(data.shape[0])

		nb_ex = data.shape[0]

		for index in range(nb_ex):
			z = data[index,:]
			index_NN = self.find_index_NN(z)
			z_NN = self.data[index_NN, :]
			if self.negativ == True and self.label[index_NN] == -1:
				label[index] = -1
			else:
				index_NN_NN = self.find_index_NN(z_NN, index_NN)
				z_NN_NN = self.data[index_NN_NN, :]

				if npl.norm(z- z_NN) < self.alpha * npl.norm(z_NN - z_NN_NN):
					self.lines1.append([z, z_NN])
					self.lines2.append([z_NN, z_NN_NN])
					label[index] = 1
				else:
					label[index] = -1
		return label

	def score(self, data, label):
		return np.mean(label == self.predict(data))

	def outlier_score(self, data, label):
		pred_label = self.predict(data)
		return tools.partial_score(label, pred_label, -1)

	def inlier_score(self, data, label):
		pred_label = self.predict(data)
		return tools.partial_score(label, pred_label, 1)

	def visualisation(self, data):
		self.predict(data)

		lc1 = mc.LineCollection(self.lines1, color="r")
		lc2 = mc.LineCollection(self.lines2, color="b")
		fig, ax = plt.subplots()
		ax.add_collection(lc1)
		ax.add_collection(lc2)
		ax.autoscale()
		ax.margins(0.1)
		tools.plot_data(self.data, self.label)
		tools.plot_data(data, self.predict(data), dec=2)
		plt.show()


	def find_index_NN(self, z, index=None, data=None):
		if data is None:
			data = self.data

		if index is not None:
			data = np.delete(data, index, axis=0)
		else:
			index = np.inf

		diff = npl.norm(z - data, axis = 1)
		index_NN = np.argmin(diff)
		return index_NN + int(index_NN >= index)
