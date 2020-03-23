import data
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.multiclass import OneVsRestClassifier
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import numpy as np
from itertools import cycle
import scipy

from sklearn.model_selection import train_test_split



#confusion matrix is basic, not as good as ROC, sensitivity







def main():
    

    train_data, train_labels, test_data, test_labels = data.load_all_data('data')
    #aclassifier = MyAdaBoostClassifier(train_data, train_labels, test_data, test_labels)
    #aclassifier.plot_roc_curve()
    #aclassifier.get_results()
    #mlp = MYMLPClassifier(train_data, train_labels, test_data, test_labels)
    #mlp.initialize_pipeline()
    svm = SVMClassifier(train_data, train_labels, test_data, test_labels)
    ada = MyAdaBoostClassifier(train_data, train_labels, test_data, test_labels)

    #svm.get_results()
    #svm.plot_roc_curve()

    mlp = MYMLPClassifier(train_data, train_labels, test_data, test_labels)

    metrics = Metrics(test_data, test_labels)
    #metrics.get_results(svm.initialize_classifier(), "SVM")
   # metrics.get_results(ada.initialize_classifier(), "ADA")
    metrics.get_results(mlp.initialize_classifier(), "MLP")




class SVMClassifier(object):


    def __init__(self, train_data, train_labels, test_data, test_labels):
        self.train_data = train_data
        self.train_labels = train_labels
        self.test_data = test_data
        self.test_labels = test_labels
        self.CLASSIFIER_TYPE = "Support Vector Machine"

    def initialize_classifier(self):
        print("Creating classifier: " + self.CLASSIFIER_TYPE)
        steps = [('scaler', StandardScaler()), ('SVM', SVC(kernel='poly'))]
        pipeline = Pipeline(steps)  # define Pipeline object
        parameters = {'SVM__C': [0.001, 0.1, 100, 10e5], 'SVM__gamma': [10, 1, 0.1, 0.01]}
        print("Grid searching for best parameters on: " + self.CLASSIFIER_TYPE)
        grid = GridSearchCV(pipeline, param_grid=parameters, cv=5)
        print("Fitting Classifier: " + self.CLASSIFIER_TYPE)
        grid.fit(self.train_data, self.train_labels)
        return grid


    def plot_roc_curve(self):

        y = label_binarize(self.train_labels, classes = [0,1,2,3,4,5,6,7,8,9])

        n_classes = y.shape[1]

        X_train, X_test, y_train, y_test = train_test_split(self.train_data, y, test_size=.36,
                                                            random_state=0)

        # Learn to predict each class against the other
        classifier = OneVsRestClassifier(SVC(kernel='linear', probability=True,
                                                 random_state=1))
        y_score = classifier.fit(X_train, y_train).decision_function(X_test)

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
        plt.title('Receiver Operating Characteristic Curve for {}'.format(self.CLASSIFIER_TYPE))
        plt.legend(loc="lower right")
        plt.show()


class MyAdaBoostClassifier:

    def __init__(self, train_data, train_labels, test_data, test_labels):
        self.train_data = train_data
        self.train_labels = train_labels
        self.test_data = test_data
        self.test_labels = test_labels
        self.CLASSIFIER_TYPE = "AdaBoost"


    def initialize_classifier(self):
        print("Creating classifier: " + self.CLASSIFIER_TYPE)
        clf = AdaBoostClassifier(n_estimators=46,learning_rate=1,random_state=3)
        print("Fitting Classifier: " + self.CLASSIFIER_TYPE)
        clf.fit(self.train_data, self.train_labels)
        print("Returning classifier: " + self.CLASSIFIER_TYPE)
        return clf





    def plot_roc_curve(self):

        y=label_binarize(self.train_labels, classes = [0,1,2,3,4,5,6,7,8,9])

        n_classes = y.shape[1]

        X_train, X_test, y_train, y_test = train_test_split(self.train_data, y, test_size=.36,
                                                            random_state=0)

        # Learn to predict each class against the other

        classifier = OneVsRestClassifier(AdaBoostClassifier(n_estimators=46,learning_rate=1,random_state=3))

        y_score = classifier.fit(X_train, y_train).decision_function(X_test)

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

        colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                     label='ROC curve of class {0} (area = {1:0.2f})'
                           ''.format(i, roc_auc[i]))

        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Some extension of Receiver operating characteristic to multi-class')
        plt.legend(loc="lower right")
        plt.show()


class MYMLPClassifier():

    def __init__(self, train_data, train_labels, test_data, test_labels):
        self.train_data = train_data
        self.train_labels = train_labels
        self.test_data = test_data
        self.test_labels = test_labels
        self.CLASSIFIER_TYPE = "MLP Neural Network"


    def initialize_classifier(self):
        print("Creating classifier: " + self.CLASSIFIER_TYPE)
        mlp = OneVsRestClassifier(MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000, alpha=1e-4,
                            solver='sgd', random_state=1,
                            learning_rate_init=.1))
        print("Fitting Classifier: " + self.CLASSIFIER_TYPE)
        mlp.fit(self.train_data, self.train_labels)

        return mlp





class Metrics:

    def __init__(self, test_data, test_labels):

        self.test_data = test_data
        self.test_labels = test_labels

    def get_results(self, clf, type):
        print("Predicting labels with: " + type)
        pred = clf.predict(self.test_data)
        print("Getting Classification Report for: " + type + "\n")
        print(metrics.classification_report(pred, self.test_labels))





if __name__ == '__main__':


    main()






