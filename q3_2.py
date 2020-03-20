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

    steps = [('scaler', StandardScaler()), ('SVM', SVC(kernel='poly'))]
    pipeline = Pipeline(steps)  # define Pipeline object

    print(pipeline)

    parameters = {'SVM__C': [0.001, 0.1, 100, 10e5], 'SVM__gamma': [10, 1, 0.1, 0.01]}

    grid = GridSearchCV(pipeline, param_grid=parameters, cv=5)

    print("Fitting")
    grid.fit(train_data, train_labels)

    print("Testing")

    score = grid.score(test_data, test_labels)

    print(score)

    print("Done")


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

        steps = [('scaler', StandardScaler()), ('SVM', SVC(kernel='poly'))]
        pipeline = Pipeline(steps)  # define Pipeline object
        parameters = {'SVM__C': [0.001, 0.1, 100, 10e5], 'SVM__gamma': [10, 1, 0.1, 0.01]}
        grid = GridSearchCV(pipeline, param_grid=parameters, cv=5)
        grid.fit(self.train_data, self.train_labels)

        return grid


    def get_results(self):

        pipeline = self.__initializePipeline()

        pipeline.

if __name__ == '__main__':


    main()






