import numpy as np
import numpy.linalg

import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D

import pandas as pd

import random

import sklearn.model_selection as skms



def import_datas_cardfraud(sampling=1, test_ratio=0.5, full_split=True):
    """
    Args:
        sampling (float, optional): Ratio (between 0 and 1) of data used.
    
    Returns:
        ndarray: X, x_train, x_test: datas descriptor
        ndarray: Y, y_train, y_test: datas label
    """

    ### Données credit card : le fichier Datasets ne doit pas avoir été bougé.
    print("Import des données Credit Card Fraud Detection...")
    creditfraud = pd.read_csv("Datasets/Credit Card Fraud Detection/creditcard.csv")
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

    if full_split:
    # Split des données :
        x_train, x_test, y_train, y_test = skms.train_test_split(X, Y, test_size=test_ratio) #, random_state=42)
        x_train = x_train[:int(sampling * x_train.shape[0]), :]
        x_test = x_test[:int(sampling * x_test.shape[0]), :]
        y_train = y_train[:int(sampling * len(y_train))]
        y_test = y_test[:int(sampling * len(y_test))]

        X = np.vstack((x_train, x_test))
        Y = np.hstack((y_train, y_test))
        print("Après sampling :\nNombre de données positive : {}\nNombre de données négatives : {}\nPourcentage négative sur positive : {}%".format(np.sum(Y == 1), np.sum(Y == -1), 100 * np.sum(Y == -1) / np.sum(Y == 1)))

        return X, Y, x_train, x_test, y_train, y_test
    else:
        X_sampled, X_reste, Y_sampled, Y_reste = skms.train_test_split(X, Y, train_size=sampling)

        x_train, x_test, y_train, y_test = skms.train_test_split(X_sampled, Y_sampled, test_size=test_ratio)
        x_test = np.vstack((x_test, X_reste[Y_reste == -1]))
        y_test = np.hstack((y_test, Y_reste[Y_reste == -1]))

        print("Après sampling:\nNombre de données positive de train : {}\nNombre de données négatives de train : {}\nNombres de données positives de test : {}\nNombres de données négatives de test : {}".format(np.sum(y_train==1), np.sum(y_train==-1), np.sum(y_test==1), np.sum(y_test==-1)))
        
        X = np.vstack((x_train, x_test))
        Y = np.hstack((y_train, y_test))
        return X, Y, x_train, x_test, y_train, y_test




def partial_score(a, b, in_out=1):
    return np.sum(a + b == in_out * 2 * np.ones(len(a))) / np.sum(a == in_out)

def Accuracy(ytest, ypred):
    nber_true_pos = np.sum(ytest + ypred == 1 * 2 * np.ones(len(ytest)))
    nber_true_neg = np.sum(ytest + ypred == -1 * 2 * np.ones(len(ytest)))
    return 100*(nber_true_pos+nber_true_neg)/(len(ytest))

def plot_data(data, labels=None, dec=0):
    """
    Affiche des donnees 2D
    :param data: matrice des donnees 2d
    :param labels: vecteur des labels (discrets)
    :return:
    """
    cols, marks = ["red", "blue", "green", "orange", "black", "cyan"], [".", "+", "*", "o", "x", "^"]
    if labels is None:
        plt.scatter(data[:, 0], data[:, 1], marker="x")
        return
    for i, l in enumerate(sorted(list(set(labels.flatten())))):
        # plt.scatter(data[labels == l, 0], data[labels == l, 1], c=cols[dec+i], marker=marks[dec+i])
        plt.scatter(data[labels == l, 0], data[labels == l, 1], c=cols[dec+i], marker=marks[dec+i])
        
def plot_frontiere(data,f,step=20):
    """ Trace un graphe de la frontiere de decision de f
    :param data: donnees
    :param f: fonction de decision
    :param step: pas de la grille
    :return:
    """
    grid,x,y=make_grid(data=data,step=step)
    plt.contourf(x,y,f(grid).reshape(x.shape),256)


def make_grid(data=None, xmin=-5, xmax=5, ymin=-5, ymax=5, step=20):
    """ Cree une grille sous forme de matrice 2d de la liste des points
    :param data: pour calcluler les bornes du graphe
    :param xmin: si pas data, alors bornes du graphe
    :param xmax:
    :param ymin:
    :param ymax:
    :param step: pas de la grille
    :return: une matrice 2d contenant les points de la grille
    """
    if data is not None:
        xmax, xmin, ymax, ymin = np.max(data[:, 0]), np.min(data[:, 0]), np.max(data[:, 1]), np.min(data[:, 1])
    x, y = np.meshgrid(np.arange(xmin, xmax, (xmax - xmin) * 1. / step), np.arange(ymin, ymax, (ymax - ymin) * 1. / step))
    grid = np.c_[x.ravel(), y.ravel()]
    return grid, x, y


def gen_arti(centerx=1, centery=1, sigma=0.1, nbex=1000, data_type=0, epsilon=0.02):
    """ Generateur de donnees,
        :param centerx: centre des gaussiennes
        :param centery:
        :param sigma: des gaussiennes
        :param nbex: nombre d'exemples
        :param data_type: 0: melange 2 gaussiennes, 1: melange 4 gaussiennes, 2:echequier
        :param epsilon: bruit dans les donnees
        :return: data matrice 2d des donnnes,y etiquette des donnnees
    """
    if data_type == 0:
        # melange de 2 gaussiennes
        xpos = np.random.multivariate_normal([centerx, centerx], np.diag([sigma, sigma]), int(nbex * 19 // 20))
        xneg = np.random.multivariate_normal([-centerx, -centerx], np.diag([sigma, sigma]), int(nbex // 20))
        data = np.vstack((xpos, xneg))
        y = np.hstack((np.ones(nbex * 19 // 20), -np.ones(nbex // 20)))
    if data_type == 1:
        # melange de 4 gaussiennes
        xpos = np.vstack((np.random.multivariate_normal([centerx, centerx], np.diag([sigma, sigma]), int(nbex * 19 // 40)),
                          np.random.multivariate_normal([-centerx, -centerx], np.diag([sigma, sigma]), int(nbex * 19 / 40))))
        xneg = np.vstack((np.random.multivariate_normal([-centerx, centerx], np.diag([sigma, sigma]), int(nbex // 40)),
                          np.random.multivariate_normal([centerx, -centerx], np.diag([sigma, sigma]), int(nbex / 40))))
        data = np.vstack((xpos, xneg))
        y = np.hstack((np.ones(nbex * 19 // 20), -np.ones(int(nbex // 20))))

    if data_type == 2:
        # echiquier
        data = np.reshape(np.random.uniform(-4, 4, 2 * nbex), (nbex, 2))
        y = np.ceil(data[:, 0]) + np.ceil(data[:, 1])
        y = 2 * (y % 2) - 1
    
    # un peu de bruit
    data[:, 0] += np.random.normal(0, epsilon, nbex)
    data[:, 1] += np.random.normal(0, epsilon, nbex)
    # on mélange les données
    idx = np.random.permutation((range(y.size)))
    data = data[idx, :]
    y = y[idx]
    return data, y

def gen_unbalanced(nbex_pos=100, nbex_neg=0, epsilon=0.5, ndim=2, data_type=0, sigma=0.1):

    if data_type==0:
        xpos = np.random.randn(ndim, nbex_pos)
        xpos /= np.linalg.norm(xpos, axis=0)
        xpos = np.multiply(xpos, np.random.uniform(size=(ndim, nbex_pos)))
        if nbex_neg > 0:
            xneg = np.random.randn(ndim, nbex_neg)
            xneg /= np.linalg.norm(xneg, axis=0)
            xneg = np.multiply(xneg, np.floor(10 * np.random.normal(0, sigma, (ndim, nbex_neg))))
            data=np.vstack((xpos.T, xneg.T))
            y = np.hstack((np.ones(nbex_pos), -np.ones(nbex_neg)))
        else:
            data=xpos.T
            y=np.ones(nbex_pos)

    if data_type==1:
        xpos = np.random.randn(ndim, nbex_pos)
        xpos /= np.linalg.norm(xpos, axis=0)
        if nbex_neg > 0:
            xneg = 10 * np.random.normal(0, sigma, (ndim, nbex_neg))
            data=np.vstack((xpos.T, xneg.T))
            y = np.hstack((np.ones(nbex_pos), -np.ones(nbex_neg)))
        else:
            data=xpos.T
            y=np.ones(nbex_pos)

    if data_type not in {0, 1}:
        try:
            raise ValueError("Datatype not supported")
        except ValueError as err:
            print(err)
    else:
        # un peu de bruit
        data[:, 0] += np.random.normal(0, epsilon, nbex_pos + nbex_neg)
        data[:, 1] += np.random.normal(0, epsilon, nbex_pos + nbex_neg)
        # on mélange les données
        idx = np.random.permutation((range(y.size)))
        data = data[idx, :]
        y = y[idx]
        return data, y

