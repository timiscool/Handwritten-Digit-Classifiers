import tensorflow as tf
import data
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn import svm
from sklearn.model_selection import GridSearchCV


def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')
    svm = SVMClassifier(train_data, train_labels, test_data, test_labels)

    print(svm.get_results())




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

        print("-"*5+"Testing"+"-"*5)
        score = pipeline.score(self.test_data, self.test_labels)

        print("-"*5+ "Returning score"+"-"*5)

        return score



class AdaBoostClassifier:

    def __init__(self, train_data, train_labels, test_data, test_labels):
        pass



if __name__ == '__main__':


    main()






