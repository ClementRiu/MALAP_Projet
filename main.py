import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as npl
import pandas as pd
import sklearn.model_selection as skms

import nearest_neighbours
import tools

def import_datas_cardfraud():

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
	print("Nombre de données positive : {}\nNombre de données négatives : {}\nPourcentage positive sur négative : {}%".format(np.sum(Y == 1), np.sum(Y == -1), 100 * np.sum(Y == -1) / np.sum(Y == 1)))

	# Split des données :
	x_train, x_test, y_train, y_test = skms.train_test_split(X, Y, test_size=0.33, random_state=42)
	return X, Y, x_train, x_test, y_train, y_test

if __name__ == "__main__":

	### Random datas
	# x_train, y_train = tools.gen_unbalanced(nbex_pos=50, nbex_neg=0, epsilon=0.01, data_type=1, sigma=0.5)
	# x_test, y_test = tools.gen_unbalanced(nbex_pos=5000, nbex_neg=50, epsilon=0.01, data_type=1, sigma=0.5)

	### Données credit card : le fichier Datasets ne doit pas avoir été bougé.
	
	_, _, x_train, x_test, y_train, y_test = import_datas_cardfraud()


	# # Chargement du classifieur :
	# cl = nearest_neighbours.Nearest_Neighbours(alpha=4)

	# # Transformation des données selon les labels souhaités :
	# x_train_1lab = x_train[y_train == 1]
	# y_train_1lab = y_train[y_train == 1]

	# cl.fit(x_train_1lab, y_train_1lab)

	# # Fonction spécifique au classifieur Nearest_Neighbours
	# # cl.visualisation(x_train, y_train, x_test)
	# y_predit = cl.predict(x_test)

	# print(cl.score(x_test, y_test))

	# # Fonction spécifique au classifieur Nearest_Neighbours
	# print(cl.outlier_score(x_test, y_test))

	# # Fonction spécifique au classifieur Nearest_Neighbours
	# print(cl.inlier_score(x_test, y_test))
