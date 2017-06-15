from sklearn import tree
from main import *
from tools import *
import matplotlib.pyplot as plt
import numpy as np

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
        clf = tree.DecisionTreeClassifier(criterion="gini", max_depth=11)
        clf.fit(x_train,y_train)
        y_predit = clf.predict(x_test)
        vp_score.append(tools.partial_score(y_test, y_predit, 1))
        vn_score.append(tools.partial_score(y_test, y_predit, -1))



    # vp_score = []
    # vn_score = []
    X, Y, x_train, x_test, y_train, y_test = import_datas_cardfraud(sampling=1)
    x_train_pos = x_train[y_train == +1]
    x_train_neg = x_train[y_train == -1]
    y_train_pos = y_train[y_train == +1]
    y_train_neg = y_train[y_train == -1]

    neg_max = np.sum(y_train == -1)
    value_range2 = range(0, neg_max, 10)
    for value in value_range2:
        x_train = np.vstack((x_train_pos, x_train_neg[:value]))
        y_train = np.hstack((y_train_pos, y_train_neg[:value]))
        idx = np.random.permutation((range(y_train.size)))
        x_train = x_train[idx, :]
        y_train = y_train[idx]
        clf = tree.DecisionTreeClassifier(criterion="gini", max_depth=11)
        clf.fit(x_train,y_train)
        y_predit = clf.predict(x_test)

    # is_leaves = clf.tree_.children_left == -1
    # impurity = np.dot(is_leaves, clf.tree_.impurity)


        vp_score.append(tools.partial_score(y_test, y_predit, 1))
        vn_score.append(tools.partial_score(y_test, y_predit, -1))
        print(tools.partial_score(y_test, y_predit, 1), tools.partial_score(y_test, y_predit, -1))

    # print("Profondeur de l'arbre : %d" %clf.tree_.max_depth)
    # print("Nombre de feuilles: %d" %np.sum(is_leaves))
    # print("Impuretés des feuilles: %.2f" %impurity)
    # print("Accuracy : %.4f" %clf.score(x_test,y_test))

    # true_pos = partial_score(y_test, y_pred, +1)
    # true_neg = partial_score(y_test, y_pred, -1)
    # false_neg = 0
    # false_pos = 0

    # print("\n \t Positive \t Negative")
    # print("True \t %0.4f \t %0.4f" %(100*true_pos,100*true_neg))
    # print("False \t %0.4f \t %0.4f" %(100*false_pos,100*false_neg))

    # depth_tree = range(1,23)
    # score_truepos = []
    # score_trueneg = []
    # list_impurity = []

    # with open("tree.dot", "w") as f:
    #     tree.export_graphviz(clf, out_file=f)

    # for depth in depth_tree:
    #     # print(depth)
    #     # Définition de l'arbre
    #     clf = tree.DecisionTreeClassifier(criterion="gini",max_depth=depth)

    #     # Entraînement de l'arbre
    #     clf.fit(x_train,y_train)

    #     # Prédiction
    #     y_pred = clf.predict(x_test)

    #     true_pos = partial_score(y_test, y_pred, +1)
    #     true_neg = partial_score(y_test, y_pred, -1)
    #     is_leaves = clf.tree_.children_left == -1
    #     impurity = np.dot(is_leaves, clf.tree_.impurity)
        
    #     score_truepos.append(true_pos)
    #     score_trueneg.append(true_neg)
    #     list_impurity.append(impurity)
    
    # plt.title("Score True Positive")
    # plt.plot(list(depth_tree),score_truepos)
    # plt.show()
    # plt.title("Score True Negative")
    # plt.plot(list(depth_tree),score_trueneg)
    # plt.show()
    # plt.title("Impuretés résiduelles")
    # plt.plot(list(depth_tree),list_impurity)
    # plt.show()

    print(len(vn_score))
    print(nb_essai)
    fig, ax = plt.subplots()
    plt.title("Évolution de la précision en fonction du ratio d'exemple négatif disponible")
    ax.scatter(vp_score[:nb_essai], vn_score[:nb_essai], color='r', label="Données artificielles type 1", marker="+")
    ax.scatter(vp_score[nb_essai:2*nb_essai], vn_score[nb_essai:2*nb_essai], color='b', label="Données artificielles type 2", marker="+")
    ax.scatter(vp_score[2*nb_essai:], vn_score[2*nb_essai:], color='g', label="Données Card Fraud", marker="+")
    plt.ylim(0.0, 1.05)
    plt.xlim(0.0, 1.05)
    plt.xlabel("Précision sur les positifs")
    plt.ylabel("Précision sur les négatifs")
    plt.legend(loc="lower left", borderaxespad=0.)

    for i, txt in enumerate(value_range1*2):
        ax.annotate(txt, (vp_score[i], vn_score[i]))

    for i, txt in enumerate(value_range2):
        ax.annotate(txt, (vp_score[2*nb_essai + i], vn_score[2*nb_essai + i]))
    plt.show()
