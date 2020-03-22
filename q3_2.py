import data
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.multiclass import OneVsRestClassifier

from sklearn import metrics





from sklearn.metrics import roc_auc_score
#confusing matrix is basic, not as good as ROC, sensitivity






def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')






    aclassifier = MyAdaBoostClassifier(train_data, train_labels, test_data, test_labels)
    aclassifier.get_results()







    #mlp = MYMLPClassifier(train_data, train_labels, test_data, test_labels)
    #mlp.initialize_pipeline()

    #svm = SVMClassifier(train_data, train_labels, test_data, test_labels)
    #acc = svm.get_results()







class SVMClassifier(object):


    def __init__(self, train_data, train_labels, test_data, test_labels):
        self.train_data = train_data
        self.train_labels = train_labels
        self.test_data = test_data
        self.test_labels = test_labels



    def __initializePipeline(self):

        print("Initializing Pipeline")

        steps = [('scaler', StandardScaler()), ('SVM', SVC(kernel='poly'))]
        pipeline = Pipeline(steps)  # define Pipeline object
        parameters = {'SVM__C': [0.001, 0.1, 100, 10e5], 'SVM__gamma': [10, 1, 0.1, 0.01]}
        grid = GridSearchCV(pipeline, param_grid=parameters, cv=5)
        grid.fit(self.train_data, self.train_labels)

        return grid

    def get_metrics(self, predictions, true):
        acc = accuracy_score(true, predictions)
        conf = confusion_matrix(true, predictions)
        prec = precision_score(true, predictions, average='macro', labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        recall = recall_score(true, predictions, average='macro')
        roc = 0
        return acc, conf, prec, recall, roc

    def get_results(self):

        pipeline = self.__initializePipeline()

        print("-"*5+"Fitting"+"-"*5)
        pipeline.fit(self.train_data, self.train_labels)

        pred = pipeline.predict(self.test_data)

        acc, conf, prec, recall, roc = self.get_metrics(pred, self.test_labels)


        print("Accuracy: " + str(acc))
        print("Accuracy: " + str(prec))
        print("Accuracy: " + str(recall))

        print("Accuracy: " + str(roc))



        return acc







class MyAdaBoostClassifier:

    def __init__(self, train_data, train_labels, test_data, test_labels):
        self.train_data = train_data
        self.train_labels = train_labels
        self.test_data = test_data
        self.test_labels = test_labels


    def __initialize_pipeline(self):

        print("Creating classifier")
        clf = AdaBoostClassifier(n_estimators=46,learning_rate=1,random_state=3)
        print("Fitting Classifier")
        clf.fit(self.train_data, self.train_labels)

        print("Returning classifier")
        return clf


    def get_metrics(self, predictions, true):
        acc = accuracy_score(true, predictions)
        conf = confusion_matrix(true, predictions)
        prec = precision_score(true, predictions, average='macro')
        recall = recall_score(true, predictions, average='macro')
        roc = 0

        return acc, conf, prec, recall, roc

    def get_results(self):


        fitted_clf = self.__initialize_pipeline()

        print("Predicting")
        print(len(self.test_data))
        pred = fitted_clf.predict(self.test_data)
        print("Scoring")
        print(fitted_clf.score(self.test_data, self.test_labels))

        acc, conf, prec, recall, roc = self.get_metrics(pred, self.test_labels)

        print("Accuracy score is: " + str(acc))
        print("Precision score is: " + str(prec))
        print("Recall score is: " + str(recall))
        print("ROC score is: " + str(roc))

        print(metrics.classification_report(pred, self.test_labels))












class MYMLPClassifier():

    def __init__(self, train_data, train_labels, test_data, test_labels):
        self.train_data = train_data
        self.train_labels = train_labels
        self.test_data = test_data
        self.test_labels = test_labels




    def get_metrics(self, predictions, true):
        acc = accuracy_score(true, predictions)
        conf = confusion_matrix(true, predictions)
        prec = precision_score(true, predictions, average='macro',labels=[0,1,2,3,4,5,6,7,8,9])
        recall = recall_score(true, predictions, average='macro')
        roc =0

        return acc, conf, prec, recall, roc




    def initialize_pipeline(self):



        print("Encoding")

        enc = OneHotEncoder(handle_unknown='ignore')

        enc.fit(self.train_data, self.train_labels)



        mlp = OneVsRestClassifier(MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000, alpha=1e-4,
                            solver='sgd', random_state=1,
                            learning_rate_init=.1))


        mlp.fit(self.train_data, self.train_labels)

        pred = mlp.predict(self.test_data)

        acc, conf, prec,rec,roc= self.get_metrics(self.test_labels, pred)

        print("Accuracy score is: " + str(acc))
        print("Precision score is: " + str(rec))
        print("Recall score is: " + str(rec))
        print("ROC score is: " + str(roc))






        #print("training score : " + str(mlp.score(self.train_data, self.train_labels)))

        #print("testing score: " + str(mlp.score(self.test_data, self.test_labels)))





if __name__ == '__main__':


    main()






