import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn import preprocessing

dataset = load_breast_cancer()

x = dataset.data
y = dataset.target
y = y.reshape(y.shape[0], 1)
classes = dataset.target_names.shape[0]


def confusion_matrix(input, target):
    true_positive = 0
    true_negative = 0
    flase_positive = 0
    flase_negative = 0
    for i in range(input.shape[0]):
        if input[i] >= 0.5:
            input[i] = 1
        else:
            input[i] = 0

    for i in range(target.shape[0]):
        if input[i] == target[i]:
            if input[i] == 1:
                true_positive += 1
            else:
                true_negative += 1
        elif input[i] != target[i]:
            if input[i] == 0:
                flase_negative += 1
            else:
                flase_positive += 1
    print("Confusion Matrix: ", np.array([[true_positive, flase_positive], [flase_negative, true_negative]]))
    print("True Positive = ", true_positive)
    print("False Positive = ", flase_positive)
    print("True Negative = ", true_negative)
    print("False Negative = ", flase_negative)


def train_test_split(input, target, ratio):
    shuffle_indices = np.random.permutation(input.shape[0])
    test_size = int(input.shape[0] * ratio)
    train_indices = shuffle_indices[:test_size]
    test_indices = shuffle_indices[test_size:]
    xTrain = input[train_indices]
    yTrain = target[train_indices]
    xTest = input[test_indices]
    yTest = target[test_indices]
    return xTrain, yTrain, xTest, yTest


class FeatureScaling:
    def __init__(self, data, type='Normalization'):
        self.data = data
        self.type = type

    def Scaling(self):

        if self.type == 'Normalization':
            for i in range(0, self.data.shape[1]):
                self.data[:, i] = (self.data[:, i] - min(self.data[:, i])) / (
                        max(self.data[:, i]) - min(self.data[:, i]))
            return self.data

        elif self.type == 'Standardization':
            for i in range(0, self.data.shape[1]):
                self.data[:, i] = (self.data[:, i] - np.mean(self.data[:, i])) / (np.var(self.data[:, i]))
            return self.data

        elif self.type == 'Mean_Normalization':
            for i in range(0, self.data.shape[1]):
                self.data[:, i] = (self.data[:, i] - np.mean(self.data[:, i])) / (
                        max(self.data[:, i]) - min(self.data[:, i]))
            return self.data


class NearestNeighbourError(Exception):
    pass


class KNN:
    def __init__(self, data, target, neighbour, classes, norm='L2'):
        self.data = data
        self.target = target
        self.neighbour = neighbour
        self.norm = norm
        self.normValue = self.extract_norm()
        self.classes = classes

    def distance(self, input):
        self.input = input
        self.output_list = {}
        for i in range(0, self.data.shape[0]):
            self.diffrence = np.abs(self.data[i]-self.input)
            self.diffrence = self.diffrence.reshape(self.data.shape[1],1)
            self.norm_out = np.power(self.diffrence, self.normValue)
            self.output = np.power(np.sum(self.norm_out), 1/self.normValue)
            self.output_list[self.output] = i

        return self.output_list

    def extract_norm(self):
        self.norm_value = int(self.norm[-1])
        return self.norm_value

    def nearest_neighbour_output(self, input):
        self.input = input
        nearestPoints = []
        sortedList = sorted(self.distance(input).items())
        for i in range(self.neighbour):
            nearestPoints.append(sortedList[i][1])

        print(nearestPoints)
        return nearestPoints

    def majority_class(self,input):
        self.input = input
        self.classList = [0]*self.classes
        for i in self.nearest_neighbour_output(self.input):
            self.classList[int(self.target[i])] += 1

        return self.classList.index(max(self.classList))


    def predict(self, input):
        self.input = input
        return self.majority_class(self.input)


xTrain, yTrain, xTest, yTest = train_test_split(x,y,0.8)
featureScaling = FeatureScaling(x)
featureScaling.Scaling()

result = []

knn = KNN(xTrain, yTrain, 3, classes)
for i in range(xTest.shape[0]):
    pred = knn.predict(xTest[i])
    result.append(pred)

confusion_matrix(np.array(result), yTest)





