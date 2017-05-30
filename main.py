import matplotlib.pyplot as plt

import numpy as np
import numpy.linalg as npl

import pandas as pd

from sklearn.metrics import f1_score, roc_auc_score, roc_curve, auc, precision_recall_curve, average_precision_score
import sklearn.model_selection as skms
import sklearn.svm as sksvm

import nearest_neighbours
import tools



def import_datas_cardfraud(sampling=1):
	"""
	Args:
	    sampling (float, optional): Ratio (between 0 and 1) of data used.
	
	Returns:
	    ndarray: X, x_train, x_test: datas descriptor
		ndarray: Y, y_train, y_test: datas label
	"""

	### Données credit card : le fichier Datasets ne doit pas avoir été bougé.
	print("Import des données Credit Card Fraud Detection...")
	creditfraud = pd.read_csv("Datasets/Credit_Card_Fraud_Detection/creditcard.csv")
	X_col_list = [
	"V1",
	"V2",
	"V3",
	"V4",
	"V5",
	"V6",
	"V7",
	"V8",
	"V9",
	"V10",
	"V11",
	"V12",
	"V13",
	"V14",
	"V15",
	"V16",
	"V17",
	"V18",
	"V19",
	"V20",
	"V21",
	"V22",
	"V23",
	"V24",
	"V25",
	"V26",
	"V27",
	"V28",
	"Amount",
	]
	X = pd.DataFrame.as_matrix(creditfraud[X_col_list])
	Y = pd.DataFrame.as_matrix(creditfraud["Class"])

	# Transformation des résultats de façon à ce que la classe positive soit +1 et la classe négative -1 :
	# Attention, ce passage est spécifique à la structure des données fournies (une fonction plus générale va arriver plus tard)
	Y[Y == 1] = -1
	Y[Y == 0] = 1
	print("Avant sampling :\nNombre de données positive : {}\nNombre de données négatives : {}\nPourcentage négative sur positive : {}%".format(np.sum(Y == 1), np.sum(Y == -1), 100 * np.sum(Y == -1) / np.sum(Y == 1)))

	# Split des données :
	x_train, x_test, y_train, y_test = skms.train_test_split(X, Y, test_size=0.33, random_state=42)
	x_train = x_train[:int(sampling * x_train.shape[0]), :]
	x_test = x_test[:int(sampling * x_test.shape[0]), :]
	y_train = y_train[:int(sampling * len(y_train))]
	y_test = y_test[:int(sampling * len(y_test))]

	X = np.vstack((x_train, x_test))
	Y = np.hstack((y_train, y_test))
	print("Après sampling :\nNombre de données positive : {}\nNombre de données négatives : {}\nPourcentage négative sur positive : {}%".format(np.sum(Y == 1), np.sum(Y == -1), 100 * np.sum(Y == -1) / np.sum(Y == 1)))

	return X, Y, x_train, x_test, y_train, y_test

if __name__ == "__main__":

	### Random datas
	x_train, y_train = tools.gen_unbalanced(nbex_pos=500, nbex_neg=10, epsilon=0.01, data_type=1, sigma=0.5)
	x_test, y_test = tools.gen_unbalanced(nbex_pos=5000, nbex_neg=50, epsilon=0.01, data_type=1, sigma=0.5)

	### Données credit card : le fichier Datasets ne doit pas avoir été bougé.
	
	# _, Y, x_train, x_test, y_train, y_test = import_datas_cardfraud(0.2)
	nbex = len(Y)

	cl = sksvm.OneClassSVM()

	param_grid = {
		"nu":[0.01, 0.1, 0.3, 0.5, 0.7, 0.9],
		"gamma":[1/4/nbex, 1/2/nbex, 1/nbex, 2/nbex, 10/nbex, 20/nbex],
	}

	cl = skms.GridSearchCV(cl, param_grid, scoring="roc_auc", n_jobs=8)

	cl.fit(x_train, y_train)
	print(cl.get_params())
	y_predit = cl.predict(x_test)
	# print(cl.score(x_train, y_train))
	# print(cl.score(x_test, y_test))

	print(tools.partial_score(y_test, y_predit, 1))
	print(tools.partial_score(y_test, y_predit, -1))



	# Données aléatoires
	# Chargement du classifieur :
	# cl = nearest_neighbours.Nearest_Neighbours(alpha=1)

	# # Transformation des données selon les labels souhaités :
	# x_train_1lab = x_train[y_train == 1]
	# y_train_1lab = y_train[y_train == 1]

	# cl.fit(x_train_1lab, y_train_1lab)

	# # Fonction spécifique au classifieur Nearest_Neighbours
	# # cl.visualisation(x_train, y_train, x_test)
	# y_predit = cl.predict(x_test)

	# print(np.sum(y_test == -1))
	# print(np.sum(y_predit == -1))
	# print(tools.partial_score(y_test, y_predit, 1))
	# print(tools.partial_score(y_test, y_predit, -1))

	# print(cl.score(x_test, y_test))

	# Fonction spécifique au classifieur Nearest_Neighbours
	# print(cl.outlier_score(x_test, y_test))

	# # Fonction spécifique au classifieur Nearest_Neighbours
	# print(cl.inlier_score(x_test, y_test))
