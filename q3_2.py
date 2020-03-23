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
#confusing matrix is basic, not as good as ROC, sensitivity






def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')
    aclassifier = MyAdaBoostClassifier(train_data, train_labels, test_data, test_labels)
    aclassifier.get_results()
    mlp = MYMLPClassifier(train_data, train_labels, test_data, test_labels)
    mlp.initialize_pipeline()
    svm = SVMClassifier(train_data, train_labels, test_data, test_labels)
    svm.get_results()







class SVMClassifier(object):


    def __init__(self, train_data, train_labels, test_data, test_labels):
        self.train_data = train_data
        self.train_labels = train_labels
        self.test_data = test_data
        self.test_labels = test_labels
        self.CLASSIFIER_TYPE = "Support Vector Machine"

    def __initializePipeline(self):


        print("Creating classifier: " + self.CLASSIFIER_TYPE)
        steps = [('scaler', StandardScaler()), ('SVM', SVC(kernel='poly'))]
        pipeline = Pipeline(steps)  # define Pipeline object
        parameters = {'SVM__C': [0.001, 0.1, 100, 10e5], 'SVM__gamma': [10, 1, 0.1, 0.01]}
        print("Grid searching for best parameters on: " + self.CLASSIFIER_TYPE)
        grid = GridSearchCV(pipeline, param_grid=parameters, cv=5)

        print("Fitting Classifier: " + self.CLASSIFIER_TYPE)
        grid.fit(self.train_data, self.train_labels)

        return grid

    def get_results(self):

        clf = self.__initializePipeline()

        print("Predicting labels with: " + self.CLASSIFIER_TYPE)
        pred = clf.predict(self.test_data)

        print("Getting Classification Report for: " + self.CLASSIFIER_TYPE + "\n")
        print(metrics.classification_report(pred, self.test_labels))




class MyAdaBoostClassifier:

    def __init__(self, train_data, train_labels, test_data, test_labels):
        self.train_data = train_data
        self.train_labels = train_labels
        self.test_data = test_data
        self.test_labels = test_labels
        self.CLASSIFIER_TYPE = "AdaBoost"


    def __initialize_pipeline(self):

        print("Creating classifier: " + self.CLASSIFIER_TYPE)
        clf = AdaBoostClassifier(n_estimators=46,learning_rate=1,random_state=3)
        print("Fitting Classifier: " + self.CLASSIFIER_TYPE)
        clf.fit(self.train_data, self.train_labels)

        print("Returning classifier: " + self.CLASSIFIER_TYPE)
        return clf


    def get_results(self):

        fitted_clf = self.__initialize_pipeline()

        print("Predicting labels with: " + self.CLASSIFIER_TYPE)
        pred = fitted_clf.predict(self.test_data)
        print("Getting Classification Report for: " + self.CLASSIFIER_TYPE + "\n")
        print(metrics.classification_report(pred, self.test_labels))




class MYMLPClassifier():

    def __init__(self, train_data, train_labels, test_data, test_labels):
        self.train_data = train_data
        self.train_labels = train_labels
        self.test_data = test_data
        self.test_labels = test_labels
        self.CLASSIFIER_TYPE = "MLP Neural Network"


    def initialize_pipeline(self):


        print("Creating classifier: " + self.CLASSIFIER_TYPE)
        mlp = OneVsRestClassifier(MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000, alpha=1e-4,
                            solver='sgd', random_state=1,
                            learning_rate_init=.1))


        print("Fitting Classifier: " + self.CLASSIFIER_TYPE)
        mlp.fit(self.train_data, self.train_labels)

        print("Predicting labels with: " + self.CLASSIFIER_TYPE)
        pred = mlp.predict(self.test_data)


        print("Getting Classification Report for: " + self.CLASSIFIER_TYPE + "\n")
        print(metrics.classification_report(pred, self.test_labels))





if __name__ == '__main__':


    main()






