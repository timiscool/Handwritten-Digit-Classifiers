import data
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder



estimations = []
lol = []

def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')
    svm = SVMClassifier(train_data, train_labels, test_data, test_labels)
    print(svm.get_results())

    aclassifier = MyAdaBoostClassifier(train_data, train_labels, test_data, test_labels)
    aclassifier.get_results()

    mlp = MYMLPClassifier(train_data, train_labels, test_data, test_labels)

    mlp.initialize_pipeline()







class SVMClassifier(object):
    '''
    K Nearest Neighbor classifier
    '''

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


    def get_results(self):

        pipeline = self.__initializePipeline()

        print("-"*5+"Fitting"+"-"*5)
        pipeline.fit(self.train_data, self.train_labels)

        pred = pipeline.predict(self.test_data)
        print("SVM think its: " + str(pred[0]))
        print(len(pred))


        print("-"*5+"Testing"+"-"*5)
        score = pipeline.score(self.test_data, self.test_labels)

        print("-"*5+ "Returning score"+"-"*5)

        return score



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


    def get_results(self):


        fitted_clf = self.__initialize_pipeline()

        print("Predicting")
        pred = fitted_clf.predict(self.test_data)
        print("ADA think its: " + str(pred[0]))
        print(len(pred))



        print("Scoring")
        print(fitted_clf.score(self.test_data, self.test_labels))






class MYMLPClassifier():

    def __init__(self, train_data, train_labels, test_data, test_labels):
        self.train_data = train_data
        self.train_labels = train_labels
        self.test_data = test_data
        self.test_labels = test_labels


    def initialize_pipeline(self):


        enc = OneHotEncoder(handle_unknown='ignore')

        enc.fit(self.train_data, self.train_labels)



        mlp = MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000, alpha=1e-4,
                            solver='sgd', random_state=1,
                            learning_rate_init=.1)


        mlp.fit(self.train_data, self.train_labels)

        pred = mlp.predict(self.test_data)

        print("NN think its: " + str(pred[0]))
        print(len(pred))


        #print("training score : " + str(mlp.score(self.train_data, self.train_labels)))

        #print("testing score: " + str(mlp.score(self.test_data, self.test_labels)))






if __name__ == '__main__':


    main()






