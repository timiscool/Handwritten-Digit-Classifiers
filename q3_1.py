'''
Question 3.1 Skeleton Code

Here you should implement and evaluate the k-NN classifier.
'''

import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from collections import Counter


class KNearestNeighbor(object):
    '''
    K Nearest Neighbor classifier
    '''

    def __init__(self, train_data, train_labels):
        self.train_data = train_data
        self.train_norm = (self.train_data**2).sum(axis=1).reshape(-1,1)
        self.train_labels = train_labels

    def l2_distance(self, test_point):
        '''
        Compute L2 distance between test point and each training point
        
        Input: test_point is a 1d numpy array
        Output: dist is a numpy array containing the distances between the test point and each training point
        '''
        # Process test point shape
        test_point = np.squeeze(test_point)
        if test_point.ndim == 1:
            test_point = test_point.reshape(1, -1)
        assert test_point.shape[1] == self.train_data.shape[1]

        # Compute squared distance
        test_norm = (test_point**2).sum(axis=1).reshape(1,-1)
        dist = self.train_norm + test_norm - 2*self.train_data.dot(test_point.transpose())
        return np.squeeze(dist)

    def query_knn(self, test_point, k):
        '''
        Query a single test point using the k-NN algorithm

        You should return the digit label provided by the algorithm
        '''




        distances = self.l2_distance(test_point) # get distances
        sorted_index = np.argpartition(distances, k)[:k] # get indencies of distances


        labels = self.train_labels[sorted_index] # get the labels, based on indencies
        (label, count) = np.unique(labels, return_counts=True) # get the


        c = Counter(labels)

        label, count = c.most_common()[0]
        digit = label
        return digit





        #return value[np.argmax(count)]

def cross_validation(train_data, train_labels, k_range=np.arange(1,16)):
    '''
    Perform 10-fold cross validation to find the best value for k

    Note: Previously this function took knn as an argument instead of train_data,train_labels.
    The intention was for students to take the training data from the knn object - this should be clearer
    from the new function signature.
    '''

    avg = []

    for k in k_range:

        kf = KFold(n_splits=10)

        test_accuracy = []
        print(str(k))

        for train_index, test_index in kf.split(train_data):
            knn = KNearestNeighbor(train_data[train_index, :], train_labels[train_index])
            test_tmp = classification_accuracy(knn, k, train_data[test_index, :], train_labels[test_index])
            test_accuracy.append(test_tmp)

        avg.append(np.mean(test_accuracy))

    y = np.arange(15) + 1

    plt.plot(y[:], avg[:])

    plt.xlabel("K")
    plt.ylabel("accuracy")
    plt.title("accuracy vs K")

    plt.show()

    return avg



def classification_accuracy(knn, k, eval_data, eval_labels):
    '''
    Evaluate the classification accuracy of knn on the given 'eval_data'
    using the labels
    '''

    size = np.size(eval_labels)
    correct = 0


    for x in range(size):


        if not (knn.query_knn(eval_data[x, :], k) == eval_labels[x]):
            continue

        correct += 1

    acc = correct/float(size)

    return acc



def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')
    knn = KNearestNeighbor(train_data, train_labels)


    #print(knn.l2_distance(test_data[0]))
    # Example usage:
    #predicted_label = knn.query_knn(test_data[0], 1)
    #print(predicted_label)

    print("Accuracy of k = 1 :" + str(classification_accuracy(knn, 1, test_data, test_labels)))

    print("Accuracy of k = 15 :" + str(classification_accuracy(knn, 15, test_data, test_labels)))

    print(cross_validation(train_data,train_labels))



if __name__ == '__main__':

    main()

