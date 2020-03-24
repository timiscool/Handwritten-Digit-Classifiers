import data
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import numpy as np
from itertools import cycle
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# confusion matrix is basic, not as good as ROC, sensitivity





train_data, train_labels, test_data, test_labels = data.load_all_data('data')


def main():

    # get data, and store it to be passed around
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')
    all_data = Data(train_data, train_labels, test_data, test_labels)

    #get all my classifiers
    mlp = MYMLPClassifier(all_data)
    svm = SVMClassifier(all_data)
    ada = MyAdaBoostClassifier(all_data)
    knn = KNN(all_data)

    #load class to determine metrics
    metrics = Metrics(all_data)




    #initialize and determine metrics for AdaBoost
    grid, X_train, X_test, y_train, y_test, y_score, pred, n_classes, type = ada.initialize_classifier()
    metrics.plot_roc_curve(grid, X_train, X_test, y_train, y_test, y_score, pred, n_classes, type)

    #initialize and determine metrics for MLP Neural Network
    grid, X_train, X_test, y_train, y_test, y_score, pred, n_classes, type = mlp.initialize_classifier()
    metrics.plot_roc_curve(grid, X_train, X_test, y_train, y_test, y_score, pred, n_classes, type)

    #initialize and determine metrics for Support Vector Machine
    grid, X_train, X_test, y_train, y_test, y_score, pred, n_classes, type = svm.initialize_classifier()
    metrics.plot_roc_curve(grid, X_train, X_test, y_train, y_test, y_score, pred, n_classes, type)

    #initialize and determine metrics for K Nearest Neighbor
    grid, X_train, X_test, y_train, y_test, y_score, pred, n_classes, type = knn.initialize_classifier()
    metrics.plot_roc_curve(grid, X_train, X_test, y_train, y_test, y_score, pred, n_classes, type)




class SVMClassifier(object):
    def __init__(self, data):
        self.train_data = data.train_data
        self.train_labels = data.train_labels
        self.test_data = data.test_data
        self.test_labels = data.test_labels
        self.CLASSIFIER_TYPE = "Support Vector Machine"

    def initialize_classifier(self):
        y = label_binarize(self.train_labels, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

        n_classes = y.shape[1]

        X_train, X_test, y_train, y_test = train_test_split(self.train_data, y, test_size=.36,
                                                            random_state=0)

        steps = [('scaler', StandardScaler()), ('SVM', SVC(kernel='poly'))]
        pipeline = Pipeline(steps)

        # parameters used to tune this are below, I removed them from the grid for computation time
        # in the event that an instructor/TA runs my code.

        # parameters = {'SVM__C': [0.001, 0.1, 100, 10e5], 'SVM__gamma': [10, 1, 0.1, 0.01]}
        parameters = {'SVM__C': [0.001, 0.1, 100, 10e5], 'SVM__gamma': [10, 1, 0.1, 0.01]}
        print("Grid searching for best parameters on: " + self.CLASSIFIER_TYPE)
        grid = GridSearchCV(pipeline, param_grid=parameters, cv=5)
        print("Fitting Classifier: " + self.CLASSIFIER_TYPE)
        grid.fit(self.train_data, self.train_labels)

        # Learn to predict each class against the other
        classifier = OneVsRestClassifier(grid)
        print("Here\n")
        y_score = classifier.fit(X_train, y_train).decision_function(X_test)
        pred = grid.predict(self.test_data)
        return grid, X_train, X_test, y_train, y_test, y_score, pred, n_classes,self.CLASSIFIER_TYPE


class MyAdaBoostClassifier:
    def __init__(self, data):
        self.train_data = data.train_data
        self.train_labels = data.train_labels
        self.test_data = data.test_data
        self.test_labels = data.test_labels
        self.CLASSIFIER_TYPE = "AdaBoost"

    def initialize_classifier(self):
        y = label_binarize(self.train_labels, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        n_classes = y.shape[1]

        X_train, X_test, y_train, y_test = train_test_split(self.train_data, y, test_size=.36,
                                                            random_state=0)


        parameters = {'learning_rate': [1], 'n_estimators' : [45], 'random_state' :[3]}
        ada = AdaBoostClassifier()


        print("Grid searching for best parameters on: " + self.CLASSIFIER_TYPE)
        grid = GridSearchCV(ada, param_grid=parameters, cv=5)
        print("Fitting Grid: " + self.CLASSIFIER_TYPE)
        grid.fit(self.train_data, self.train_labels)

        # Learn to predict each class against the other
        classifier = OneVsRestClassifier(grid)
        y_score = classifier.fit(X_train, y_train).predict_proba(X_test)
        pred = grid.predict(self.test_data)


        return grid, X_train, X_test, y_train, y_test, y_score, pred, n_classes,self.CLASSIFIER_TYPE



class MYMLPClassifier():
    def __init__(self, data):
        self.train_data = data.train_data
        self.train_labels = data.train_labels
        self.test_data = data.test_data
        self.test_labels = data.test_labels
        self.CLASSIFIER_TYPE = "MLP Neural Network"

    def initialize_classifier(self):
        y = label_binarize(self.train_labels, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

        n_classes = y.shape[1]
        X_train, X_test, y_train, y_test = train_test_split(self.train_data, y, test_size=.36,
                                                            random_state=0)

        # parameters used to tune this are below, I removed them from the grid for computation time
        # in the event that an instructor/TA runs my code.

        # parameters = {'solver': ['lbfgs', 'sgd', 'adam'], 'hidden_layer_sizes': [(50,), (100,)], 'random_state': [3],
        #              'max_iter': [1000]}

        parameters = {'solver': ['adam'], 'hidden_layer_sizes': [(100,)], 'random_state': [3], 'max_iter': [1000]}
        mlp = MLPClassifier()

        print("Grid searching for best parameters on: " + self.CLASSIFIER_TYPE)
        grid = GridSearchCV(mlp, param_grid=parameters, cv=5)
        print("Fitting Grid: " + self.CLASSIFIER_TYPE)
        grid.fit(self.train_data, self.train_labels)

        # Learn to predict each class against the other
        classifier = OneVsRestClassifier(grid)
        y_score = classifier.fit(X_train, y_train).predict_proba(X_test)
        pred = grid.predict(self.test_data)

        return grid, X_train, X_test, y_train, y_test, y_score, pred, n_classes,self.CLASSIFIER_TYPE



class KNN():

    def __init__(self, data):
        self.train_data = data.train_data
        self.train_labels = data.train_labels
        self.test_data = data.test_data
        self.test_labels = data.test_labels
        self.CLASSIFIER_TYPE = "KNearest Neighbors"

    def initialize_classifier(self):
        y = label_binarize(self.train_labels, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

        n_classes = y.shape[1]

        X_train, X_test, y_train, y_test = train_test_split(self.train_data, y, test_size=.36,
                                                            random_state=0)


        parameters = {'n_neighbors': [1]}
        k = KNeighborsClassifier()

        print("Grid searching for best parameters on: " + self.CLASSIFIER_TYPE)
        grid = GridSearchCV(k, param_grid=parameters, cv=5)
        print("Fitting Grid: " + "KNeighbors")
        grid.fit(self.train_data, self.train_labels)

        # Learn to predict each class against the other
        classifier = OneVsRestClassifier(grid)
        print("Here\n")
        y_score = classifier.fit(X_train, y_train).predict_proba(X_test)
        pred = grid.predict(self.test_data)


        return grid, X_train, X_test, y_train, y_test, y_score, pred, n_classes,self.CLASSIFIER_TYPE




class Metrics:

    def __init__(self, data):
        self.train_data = data.train_data
        self.train_labels = data.train_labels
        self.test_data = data.test_data
        self.test_labels = data.test_labels

    def plot_roc_curve(self,grid, X_train, X_test, y_train, y_test, y_score, pred, n_classes,type):

        # print(metrics.classification_report(pred, self.test_labels))
        print(metrics.classification_report(self.test_labels, pred))
        print(confusion_matrix(self.test_labels, pred))
        print("Best Parameters are: " + str(grid.best_params_) + "\n")

        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        lw = 2

        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

        # Finally average it and compute AUC
        mean_tpr /= n_classes

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        # Plot all ROC curves
        plt.figure()
        plt.plot(fpr["micro"], tpr["micro"],
                 label='micro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["micro"]),
                 color='deeppink', linestyle=':', linewidth=4)

        plt.plot(fpr["macro"], tpr["macro"],
                 label='macro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["macro"]),
                 color='navy', linestyle=':', linewidth=4)

        colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'black', 'indigo', 'lime', 'gray'])
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                     label='ROC curve of class {0} (area = {1:0.2f})'
                           ''.format(i, roc_auc[i]))

        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic Curve for {}'.format(type))
        plt.legend(loc="lower right")
        plt.show()


class Data:
    def __init__(self, train_data, train_labels, test_data, test_labels):
        self.train_data = train_data
        self.train_labels = train_labels
        self.test_data = test_data
        self.test_labels = test_labels


if __name__ == '__main__':
    main()





