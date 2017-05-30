import numpy as np
import numpy.linalg
import random
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D


def partial_score(a, b, in_out=1):
    return np.sum(a + b == in_out * 2 * np.ones(len(a))) / np.sum(a == in_out)

def plot_data(data, labels=None, dec=0):
    """
    Affiche des donnees 2D
    :param data: matrice des donnees 2d
    :param labels: vecteur des labels (discrets)
    :return:
    """
    cols, marks = ["red", "blue", "green", "orange", "black", "cyan"], [".", "+", "*", "o", "x", "^"]
    print(data.shape)
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
    
    if data_type==1:
        xpos = np.random.randn(ndim, nbex_pos)
        xpos /= np.linalg.norm(xpos, axis=0)

        if nbex_neg > 0:
            xneg = np.random.randn(ndim, nbex_neg)
            xneg /= np.linalg.norm(xneg, axis=0)
            xneg += np.random.normal(scale=sigma, size=nbex_neg)
            data=np.vstack((xpos.T, xneg.T))
            y = np.hstack((np.ones(nbex_pos), -np.ones(nbex_neg)))
        else:
            data=xpos.T
            y=np.ones(nbex_pos)

    # un peu de bruit
    data += np.random.normal(0, epsilon, (nbex_pos + nbex_neg, ndim))
    # on mélange les données
    idx = np.random.permutation((range(y.size)))
    data = data[idx, :]
    y = y[idx]
    return data, y

