import matplotlib.pyplot as plt

import numpy as np
import numpy.linalg as npl

import pandas as pd

from sklearn.metrics import f1_score, roc_auc_score, roc_curve, auc, precision_recall_curve, average_precision_score
import sklearn.model_selection as skms
import sklearn.svm as sksvm

import nearest_neighbours
import tools


if __name__ == "__main__":

	vp_score = []
	vn_score = []
	value_range1 = [0, 1, 5, 10, 30, 70, 100]
	nb_essai = len(value_range1)
	pos = 100

	for datatype in [0, 1]:
		for neg in value_range1:
			print(neg)
			# ### Random datas
			x_train, y_train = tools.gen_unbalanced(nbex_pos=pos, nbex_neg=neg, epsilon=0.01, data_type=datatype, sigma=0.5)
			x_test, y_test = tools.gen_unbalanced(nbex_pos=1000, nbex_neg=100, epsilon=0.01, data_type=datatype, sigma=0.5)
			Y = np.hstack((y_train, y_test))

	# 		### Données credit card : le fichier Datasets ne doit pas avoir été bougé.

	# 		# x_train_pos = x_train[y_train == +1]
	# 		# x_train_neg = x_train[y_train == -1]
	# 		# y_train_pos = y_train[y_train == +1]
	# 		# y_train_neg = y_train[y_train == -1]

	# 		# print(y_train_neg[:int(len(y_train_neg)*neg/100)])
	# 		# x_train = np.vstack((x_train_pos, x_train_neg[:int(len(y_train_neg)*neg/100)]))
	# 		# y_train = np.hstack((y_train_pos, y_train_neg[:int(len(y_train_neg)*neg/100)]))
	# 		# idx = np.random.permutation((range(y_train.size)))
	# 		# x_train = x_train[idx, :]
	# 		# y_train = y_train[idx]


			# cl = sksvm.OneClassSVM()

	# 		# # nbex = len(Y)
	# 		# # param_grid = {
	# 		# # 	"nu":[0.01, 0.1, 0.3, 0.5, 0.7, 0.9],
	# 		# # 	"gamma":[1/4/nbex, 1/2/nbex, 1/nbex, 2/nbex, 10/nbex, 20/nbex],
	# 		# # }

	# 		# # cl = skms.GridSearchCV(cl, param_grid, scoring="roc_auc", n_jobs=8)

	# 		# cl.fit(x_train, y_train)
	# 		# y_predit = cl.predict(x_test)
	# 		# # print(cl.score(x_train, y_train))
	# 		# # print(cl.score(x_test, y_test))

	# 		# # print(tools.partial_score(y_test, y_predit, 1))
	# 		# print(np.sum(y_test[y_test == y_predit] == 1))
	# 		# print(tools.partial_score(y_test, y_predit, -1))



			cl = nearest_neighbours.Nearest_Neighbours(alpha=4)
			cl.fit(x_train, y_train)


			# # Fonction spécifique au classifieur Nearest_Neighbours
			# # cl.visualisation(x_test)

			y_predit = cl.predict(x_test)

			vp_score.append(tools.partial_score(y_test, y_predit, 1))
			vn_score.append(tools.partial_score(y_test, y_predit, -1))

	# 		# print(cl.score(x_test, y_test))

	# 		#  Fonction spécifique au classifieur Nearest_Neighbours
	# 		# print(cl.outlier_score(x_test, y_test))

	# 		# # Fonction spécifique au classifieur Nearest_Neighbours
	# 		# print(cl.inlier_score(x_test, y_test))
	
	# fig, ax = plt.subplots()
	# plt.title("Évolution de la précision en fonction du ratio d'exemple négatif disponible")
	# ax.scatter(vp_score[:nb_essai], vn_score[:nb_essai], color='r', label="Données artificielles type 1", marker="+")
	# ax.scatter(vp_score[nb_essai:], vn_score[nb_essai:], color='b', label="Données artificielles type 2", marker="+")
	# plt.ylim(0.45, 1.05)
	# plt.xlim(0.45, 1.05)
	# plt.xlabel("Précision sur les positifs")
	# plt.ylabel("Précision sur les négatifs")
	# plt.legend(loc="lower left", borderaxespad=0.)

	# for i, txt in enumerate(value_range*2):
	# 	ax.annotate(txt, (vp_score[i], vn_score[i]))
	# plt.show()
	




	_, Y, x_train, x_test, y_train, y_test = tools.import_datas_cardfraud(sampling=0.2, test_ratio=0.60)
	x_train_pos = x_train[y_train == +1]
	x_train_neg = x_train[y_train == -1]
	y_train_pos = y_train[y_train == +1]
	y_train_neg = y_train[y_train == -1]

	neg_max = np.sum(y_train == -1)
	value_range2 = range(0, neg_max, 10)
	for value in value_range2:
		print(value)

		x_train = np.vstack((x_train_pos, x_train_neg[:value]))
		y_train = np.hstack((y_train_pos, y_train_neg[:value]))
		idx = np.random.permutation((range(y_train.size)))
		x_train = x_train[idx, :]
		y_train = y_train[idx]

		# cl = sksvm.OneClassSVM(nu=0.5, gamma=0.068966)
		cl = nearest_neighbours.Nearest_Neighbours(alpha=3)
		cl.fit(x_train, y_train)
		y_predit = cl.predict(x_test)

		vp_score.append(tools.partial_score(y_test, y_predit, 1))
		vn_score.append(tools.partial_score(y_test, y_predit, -1))
		print(tools.partial_score(y_test, y_predit, 1), tools.partial_score(y_test, y_predit, -1))



	fig, ax = plt.subplots()
	plt.title("Évolution de la précision en fonction du ratio d'exemple négatif disponible")
	ax.scatter(vp_score[:nb_essai], vn_score[:nb_essai], color='r', label="Données artificielles type 1", marker="+")
	ax.scatter(vp_score[nb_essai:], vn_score[nb_essai:], color='b', label="Données artificielles type 2", marker="+")
	ax.scatter(vp_score[2*nb_essai:], vn_score[2*nb_essai:], color='g', label="Données Card Fraud", marker="+")
	plt.ylim(0.45, 1.05)
	plt.xlim(0.45, 1.05)
	plt.xlabel("Précision sur les positifs")
	plt.ylabel("Précision sur les négatifs")
	plt.legend(loc="lower left", borderaxespad=0.)

	for i, txt in enumerate(value_range1*2):
		ax.annotate(txt, (vp_score[i], vn_score[i]))

	for i, txt in enumerate(value_range2):
		ax.annotate(txt, (vp_score[2*nb_essai + i], vn_score[2*nb_essai + i]))
	plt.show()

	# fig, ax = plt.subplots()
	# plt.title("Évolution de la précision en fonction du ratio d'exemple négatif disponible")
	# ax.scatter(vp_score, vn_score, color='r', label="Données Card Fraud", marker="+")
	# plt.ylim(0, 1.05)
	# plt.xlim(0, 1.05)
	# plt.xlabel("Précision sur les positifs")
	# plt.ylabel("Précision sur les négatifs")
	# plt.legend(loc="lower left", borderaxespad=0.)

	# for i, txt in enumerate(value_range):
	# 	ax.annotate(txt, (vp_score[i], vn_score[i]))
	# plt.show()
	# 