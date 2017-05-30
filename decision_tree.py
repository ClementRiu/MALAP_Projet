from sklearn import tree
from main import *
from tools import *
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":

    X, Y, x_train, x_test, y_train, y_test = import_datas_cardfraud(sampling=1)

    clf = tree.DecisionTreeClassifier(criterion="gini", max_depth=None)
    clf.fit(x_train,y_train)
    y_pred = clf.predict(x_test)

    is_leaves = clf.tree_.children_left == -1
    impurity = np.dot(is_leaves, clf.tree_.impurity)

    print("Profondeur de l'arbre : %d" %clf.tree_.max_depth)
    print("Nombre de feuilles: %d" %np.sum(is_leaves))
    print("Impuretés des feuilles: %.2f" %impurity)
    print("Accuracy : %.4f" %clf.score(x_test,y_test))

    true_pos = partial_score(y_test, y_pred, +1)
    true_neg = partial_score(y_test, y_pred, -1)
    false_neg = 0
    false_pos = 0

    print("\n \t Positive \t Negative")
    print("True \t %0.4f \t %0.4f" %(100*true_pos,100*true_neg))
    print("False \t %0.4f \t %0.4f" %(100*false_pos,100*false_neg))

    depth_tree = range(1,23)
    score_truepos = []
    score_trueneg = []
    list_impurity = []

    with open("tree.dot", "w") as f:
        tree.export_graphviz(clf, out_file=f)

    for depth in depth_tree:
        # print(depth)
        # Définition de l'arbre
        clf = tree.DecisionTreeClassifier(criterion="gini",max_depth=depth)

        # Entraînement de l'arbre
        clf.fit(x_train,y_train)

        # Prédiction
        y_pred = clf.predict(x_test)

        true_pos = partial_score(y_test, y_pred, +1)
        true_neg = partial_score(y_test, y_pred, -1)
        is_leaves = clf.tree_.children_left == -1
        impurity = np.dot(is_leaves, clf.tree_.impurity)
        
        score_truepos.append(true_pos)
        score_trueneg.append(true_neg)
        list_impurity.append(impurity)
    
    plt.title("Score True Positive")
    plt.plot(list(depth_tree),score_truepos)
    plt.show()
    plt.title("Score True Negative")
    plt.plot(list(depth_tree),score_trueneg)
    plt.show()
    plt.title("Impuretés résiduelles")
    plt.plot(list(depth_tree),list_impurity)
    plt.show()
