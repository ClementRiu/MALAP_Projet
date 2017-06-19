import numpy as np
import numpy.linalg as npl
import matplotlib.collections as mc
import matplotlib.pyplot as plt

import tools

class K_NN_PC:
	def __init__(self, k=1, alpha=1):
		self.k = k
		self.alpha = alpha

	def fit(self, data, label=None):
		self.nb_ex, self.dim = data.shape
		self.data = data
		k_NN_dist_tab = []

		for index in range(self.nb_ex):
			element = data[index, :]
			k_NN_dist_tab.append(self.find_k_NN(element, indexes=[index], index_out=False))

		k_NN_dist_array = np.array(k_NN_dist_tab)
		self.delta = np.max(np.mean(k_NN_dist_array, axis=0))
		self.delta += (self.delta < 10**-10) * 10**-10

	def predict(self, data):
		data = data.reshape(-1, self.dim)
		diff = np.zeros((data.shape[0], self.data.shape[0]))		
		label = np.zeros(data.shape[0])

		nb_ex = data.shape[0]

		for index in range(nb_ex):
			element = data[index, :]
			k_NN_dist = self.find_k_NN(element, index_out=False)
			if np.mean(k_NN_dist) < self.alpha * self.delta:
				label[index] = 1
			else:
				label[index] = -1

		return label

	def predict_multi_alpha(self, data, alpha_range=None):
		if alpha_range is None:
			return self.predict(data)
		else:
			data = data.reshape(-1, self.dim)
			diff = np.zeros((data.shape[0], self.data.shape[0]))		
			labels = np.zeros((data.shape[0], len(alpha_range)))
			alpha_range = np.array(alpha_range)
			nb_ex = data.shape[0]

			for index in range(nb_ex):
				element = data[index, :]
				k_NN_dist = self.find_k_NN(element, index_out=False)
				labels[index, :] = ((np.mean(k_NN_dist) / self.delta) < alpha_range) * 2 - 1

			return labels

	def score(self, data, label):
		return np.mean(label == self.predict(data))

	def outlier_score(self, data, label):
		pred_label = self.predict(data)
		return tools.partial_score(label, pred_label, -1)

	def inlier_score(self, data, label):
		pred_label = self.predict(data)
		return tools.partial_score(label, pred_label, 1)

	def find_k_NN(self, element, indexes=None, data=None, index_out=True, dist_out=True):
		if data is None:
			data = self.data

		diff = npl.norm(element - data, axis = 1)
		index_sorted = np.argsort(diff)
		if indexes is not None:
			index_sorted = [index for index in index_sorted if index not in indexes]
		k_NN_index = index_sorted[:self.k]
		if dist_out and index_out:
			return k_NN_index, diff[k_NN_index]
		
		elif index_out:
			return k_NN_index
		
		elif dist_out:
			return diff[k_NN_index]
		
		else:
			return None




if __name__ == "__main__":
	test = 1

	if test == 1 :
		x_train, y_train = tools.gen_unbalanced(nbex_pos=1000, nbex_neg=15, epsilon=0.01, ndim=2, data_type=1, sigma=0.5)
		x_test, y_test = tools.gen_unbalanced(nbex_pos=1000, nbex_neg=100, epsilon=0.01, ndim=2, data_type=1, sigma=0.5)

	elif test == 2:
		_, _, x_train, x_test, y_train, y_test = tools.import_datas_cardfraud(sampling=0.04, test_ratio=0.40, full_split=False)

	vn_score_tab = []
	vp_score_tab = []

	# k_range = [1]
	# k_range = [2, 3, 4, 5, 6]
	k_range = [10, 20, 50, 100, 200]

	a_range = [0.01, 0.02, 0.03, 0.05, 0.07, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.8, 9, 9.5, 10, 15, 30, 45, 60, 75, 90, 100]
	
	# for a in a_range:
	for k in k_range:
		print(k)
		cl = K_NN_PC(k=k)
		cl.fit(x_train)
		y_predit = cl.predict_multi_alpha(x_test, a_range)

		for a_index in range(len(a_range)):
			vp_score = tools.partial_score(y_test, y_predit[:, a_index], 1)
			vn_score = tools.partial_score(y_test, y_predit[:, a_index], -1)

			vp_score_tab.append(vp_score)
			vn_score_tab.append(vn_score)

	print1b1 = False
	if print1b1:
		for k_index in range(len(k_range)):
			plt.figure()
			plt.semilogx(a_range, vn_score_tab[k_index * len(a_range) : (k_index + 1) * len(a_range)], "--", label=r"Vrai négatif : k = "+str(k_range[k_index]), color=tools.colors[k_index])
			plt.semilogx(a_range, vp_score_tab[k_index * len(a_range) : (k_index + 1) * len(a_range)], label=r"Vrai positif : k = "+str(k_range[k_index]), color=tools.colors[k_index])
			plt.xlabel(r"Valeur de $\alpha$")
			plt.ylabel("Précision")
			plt.title(r"Évolution de la précision en fonction du nombre de plus proche voisin et du coefficient $\alpha$.", wrap=True)
			plt.legend()
			plt.show()
	else:
		plt.figure()
		for k_index in range(len(k_range)):
			plt.semilogx(a_range, vn_score_tab[k_index * len(a_range) : (k_index + 1) * len(a_range)], "--", label=r"Vrai négatif : k = "+str(k_range[k_index]), color=tools.colors[k_index])
			plt.semilogx(a_range, vp_score_tab[k_index * len(a_range) : (k_index + 1) * len(a_range)], label=r"Vrai positif : k = "+str(k_range[k_index]), color=tools.colors[k_index])
		plt.xlabel(r"Valeur de $\alpha$")
		plt.ylabel("Précision")
		plt.title(r"Évolution de la précision en fonction du nombre de plus proche voisin et du coefficient $\alpha$.", wrap=True)
		plt.legend()
		plt.show()
