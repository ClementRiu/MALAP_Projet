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
	test = 2

	if test == 1 :
		x_train, y_train = tools.gen_unbalanced(nbex_pos=1000, nbex_neg=100, epsilon=0.01, ndim=2, data_type=1, sigma=0.5)
		x_test, y_test = tools.gen_unbalanced(nbex_pos=1000, nbex_neg=100, epsilon=0.01, ndim=2, data_type=1, sigma=0.5)

	elif test == 2:
		_, _, x_train, x_test, y_train, y_test = tools.import_datas_cardfraud(sampling=0.04, test_ratio=0.40, full_split=False)

	vn_score_tab = []
	vp_score_tab = []

	k_range = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 150, 200, 250, 300, ]
	a_range = [0.001, 0.01, 0.05, 0.1, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
	
	for a in a_range:
		for k in k_range:
			print(k)
			cl = K_NN_PC(k=k, alpha=a)
			cl.fit(x_train)

			y_predit = cl.predict(x_test)

			vp_score = tools.partial_score(y_test, y_predit, 1)
			vn_score = tools.partial_score(y_test, y_predit, -1)

			vp_score_tab.append(vp_score)
			vn_score_tab.append(vn_score)

	for a_index in range(len(a_range)):
		plt.figure()
		plt.plot(k_range, vn_score_tab[a_index : a_index+len(k_range)], "--", label=r"Vrai négatif : $\alpha$ = "+str(a_range[a_index]))
		plt.plot(k_range, vp_score_tab[a_index : a_index+len(k_range)], label=r"Vrai positif : $\alpha$ = "+str(a_range[a_index]))
	# plt.plot(a_range*len(k_range), vp_score_tab, label="Vrai positif")
	# plt.plot(a_range*len(k_range), vn_score_tab, label="Vrai négatif")
		plt.xlabel("Nombre k de voisins")
		plt.ylabel("Précision")
		plt.title(r"Évolution de la précision en fonction du nombre de plus proche voisin et du coefficient $\alpha$.", wrap=True)
		plt.legend()
		plt.show()
