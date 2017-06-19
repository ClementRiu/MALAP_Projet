import numpy as np
import numpy.linalg as npl

import matplotlib.collections as mc
import matplotlib.pyplot as plt

import pandas as pd

from sklearn.metrics import f1_score, roc_auc_score, roc_curve, auc, precision_recall_curve, average_precision_score
import sklearn.model_selection as skms
import sklearn.svm as sksvm


import tools



if __name__ == "__main__":
	test = 3

	if test == 1 :
		x_train, y_train = tools.gen_unbalanced(nbex_pos=1000, nbex_neg=15, epsilon=0.01, ndim=2, data_type=1, sigma=0.5)
		x_test, y_test = tools.gen_unbalanced(nbex_pos=1000, nbex_neg=100, epsilon=0.01, ndim=2, data_type=1, sigma=0.5)

	elif test == 2:
		_, _, x_train, x_test, y_train, y_test = tools.import_datas_cardfraud(sampling=0.04, test_ratio=0.40, full_split=False)

	elif test == 3:
		_, _, x_train, x_test, y_train, y_test = tools.import_datas_arrhythmia(test_ratio=0.40)

	vn_score_tab = []
	vp_score_tab = []

	nu_range = np.logspace(-3, 0, 10)
	gamma_range = np.logspace(-2, 0, 6)
	
	
	# for a in a_range:
	for gamma in gamma_range:
		print(gamma)
		for nu in nu_range:
			print(nu)
			cl = sksvm.OneClassSVM(gamma=gamma, nu=nu)
			cl.fit(x_train)
			y_predit = cl.predict(x_test)

			vp_score = tools.partial_score(y_test, y_predit, 1)
			vn_score = tools.partial_score(y_test, y_predit, -1)

			vp_score_tab.append(vp_score)
			vn_score_tab.append(vn_score)

	print1b1 = False
	if print1b1:
		for g_range in range(len(gamma_range)):
			plt.figure()
			plt.semilogx(nu_range, vn_score_tab[g_range * len(nu_range) : (g_range + 1) * len(nu_range)], "--", label=r"Vrai négatif : $\gamma$ = " + str(gamma_range[g_range]), color=tools.colors[g_range])
			plt.semilogx(nu_range, vp_score_tab[g_range * len(nu_range) : (g_range + 1) * len(nu_range)], label=r"Vrai positif : $\gamma$ = " + str(gamma_range[g_range]), color=tools.colors[g_range])
			plt.xlabel(r"Valeur de $\nu$")
			plt.ylabel("Précision")
			plt.title(r"Évolution de la précision en fonction des coefficients $\gamma$ et $\nu$.", wrap=True)
			plt.legend()
			plt.show()
	else:
		plt.figure()
		for g_range in range(len(gamma_range)):
			plt.semilogx(nu_range, vn_score_tab[g_range * len(nu_range) : (g_range + 1) * len(nu_range)], "--", label=r"Vrai négatif : $\gamma$ = " + str(gamma_range[g_range]), color=tools.colors[g_range])
			plt.semilogx(nu_range, vp_score_tab[g_range * len(nu_range) : (g_range + 1) * len(nu_range)], label=r"Vrai positif : $\gamma$ = " + str(gamma_range[g_range]), color=tools.colors[g_range])
		plt.xlabel(r"Valeur de $\nu$")
		plt.ylabel("Précision")
		plt.title(r"Évolution de la précision en fonction des coefficients $\gamma$ et $\nu$.", wrap=True)
		plt.legend()
		plt.show()
