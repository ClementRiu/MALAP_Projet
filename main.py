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

	### Random datas
	# x_train, y_train = tools.gen_unbalanced(nbex_pos=500, nbex_neg=10, epsilon=0.01, data_type=0, sigma=0.5)
	# x_test, y_test = tools.gen_unbalanced(nbex_pos=10000, nbex_neg=100, epsilon=0.01, data_type=0, sigma=0.5)
	# Y = np.hstack((y_train, y_test))

	### Données credit card : le fichier Datasets ne doit pas avoir été bougé.
	_, Y, x_train, x_test, y_train, y_test = tools.import_datas_cardfraud(0.2)

	# cl = sksvm.OneClassSVM()

	# nbex = len(Y)
	# param_grid = {
	# 	"nu":[0.01, 0.1, 0.3, 0.5, 0.7, 0.9],
	# 	"gamma":[1/4/nbex, 1/2/nbex, 1/nbex, 2/nbex, 10/nbex, 20/nbex],
	# }

	# cl = skms.GridSearchCV(cl, param_grid, scoring="roc_auc", n_jobs=8)

	# cl.fit(x_train, y_train)
	# y_predit = cl.predict(x_test)
	# # print(cl.score(x_train, y_train))
	# # print(cl.score(x_test, y_test))

	# # print(tools.partial_score(y_test, y_predit, 1))
	# print(np.sum(y_test[y_test == y_predit] == 1))
	# print(tools.partial_score(y_test, y_predit, -1))



	# Données aléatoires
	# Chargement du classifieur :
	cl = nearest_neighbours.Nearest_Neighbours(alpha=1)

	cl.fit(x_train, y_train)

	# Fonction spécifique au classifieur Nearest_Neighbours
	# cl.visualisation(x_test)
	# cl.visualisation(x_test)

	y_predit = cl.predict(x_test)

	print(tools.partial_score(y_test, y_predit, 1))
	print(tools.partial_score(y_test, y_predit, -1))

	# print(cl.score(x_test, y_test))

	#  Fonction spécifique au classifieur Nearest_Neighbours
	# print(cl.outlier_score(x_test, y_test))

	# # Fonction spécifique au classifieur Nearest_Neighbours
	# print(cl.inlier_score(x_test, y_test))
