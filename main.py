import pandas as pd
import numpy as np
import argparse
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.neural_network import MLPClassifier

class Masculinity():
    def __init__(self, dataURL):
        self.dataX = []
        self.dataY = []
        self.parse(dataURL)
        self.start_time = time.time()

    def parse(self, url):
        data = pd.read_csv(url, encoding='latin1')
        self.dataY = np.array(data.iloc[:,2])
        data = data.drop("q0001", axis = 1)
        data = pd.get_dummies(data)
        self.dataX = np.array(data)

    def plot(self, n, data, label):
        import matplotlib.pyplot
        fig1 = matplotlib.pyplot.figure()
        fig1.gca().scatter(range(n),data,label=label)
        fig1.gca().legend()
        matplotlib.pyplot.xlabel("Trial Number")
        matplotlib.pyplot.ylabel("Accuracy")
        matplotlib.pyplot.show()

    def plotAll(self, n, rf, svm, nn, somewhat, d):
        import matplotlib.pyplot
        fig1 = matplotlib.pyplot.figure()
        fig1.gca().scatter(range(n),rf,label="RF")
        fig1.gca().scatter(range(n),svm,label="SVM")
        fig1.gca().scatter(range(n),nn,label="NN")
        fig1.gca().scatter(range(n),somewhat,label="Somewhat")
        fig1.gca().legend()

        matplotlib.pyplot.xlabel("Trial Number")
        matplotlib.pyplot.ylabel("Accuracy")
        matplotlib.pyplot.show()

    def results(self, n, data, label):
        d = pd.DataFrame({label: data})

        import matplotlib.pyplot
        fig, ax = matplotlib.pyplot.subplots()
        fig.patch.set_visible(False)
        ax.axis('off')
        ax.axis('tight')
        desc = d.describe()
        desc = desc.drop(index=["count"])
        desc = desc.round(decimals = 3)
        desc = desc.append(pd.Series({label:time.time() - self.start_time}, name="Time"))
        ax.table(cellText=desc.values, rowLabels=desc.index, colLabels=desc.columns, loc='center')
        matplotlib.pyplot.show()

        self.plot(n, data, label)

    def resultsCompare(self, n, rfData, svcData, nnData, somewhat):
        d = pd.DataFrame({"RF": rfData, "SVM": svcData, "NN": nnData, "Somewhat": somewhat})

        import matplotlib.pyplot
        fig, ax = matplotlib.pyplot.subplots()
        fig.patch.set_visible(False)
        ax.axis('off')
        ax.axis('tight')
        desc = d.describe()
        desc = desc.drop(index=["count"])
        desc = desc.round(decimals = 3)
        ax.table(cellText=desc.values, rowLabels=desc.index, colLabels=desc.columns, loc='center')
        matplotlib.pyplot.show()

        self.plotAll(n, rfData, svcData, nnData, somewhat, d)

    def rf(self, n, estimators, criterion, depth):
        data = []
        for i in range(0, n):
            print("Trial #", i)
            self.trainX, self.testX, self.trainY, self.testY = train_test_split(self.dataX, self.dataY, test_size=0.75)
            rf = RandomForestClassifier(n_estimators = estimators, criterion=criterion, max_depth=depth)
            rf.fit(self.trainX, self.trainY.ravel())
            predictions = rf.predict(self.testX)
            correct = predictions == self.testY
            rate = np.sum(correct) / correct.size
            data.append(rate)
            print("Percentage correct for classification in RF", round(np.sum(correct) / correct.size, 2))
        self.results(n, data, 'RF')

    def svm(self, n, kernel, gamma):
        data = []
        for i in range(0, n):
            print("Trial #", i)
            self.trainX, self.testX, self.trainY, self.testY = train_test_split(self.dataX, self.dataY, test_size=0.75)
            s = svm.SVC(gamma=gamma, kernel=kernel)
            s.fit(self.trainX, self.trainY.ravel())
            predictions = s.predict(self.testX)
            correct = predictions == self.testY
            rate = np.sum(correct) / correct.size
            data.append(rate)
            print("Percentage correct for classification in SVC", round(np.sum(correct) / correct.size, 2))
        self.results(n, data, "SVM")

    def nn(self, n, activation, solver, alpha, iterations):
        data = []
        for i in range(0, n):
            print("Trial #", i)
            self.trainX, self.testX, self.trainY, self.testY = train_test_split(self.dataX, self.dataY, test_size=0.75)
            nn = MLPClassifier(hidden_layer_sizes=(100, 75, 50), activation=activation, solver=solver, alpha=alpha, max_iter=iterations)
            nn.fit(self.trainX, self.trainY.ravel())
            predictions = nn.predict(self.testX)
            correct = predictions == self.testY
            rate = np.sum(correct) / correct.size
            data.append(rate)
            print("Percentage correct for classification in NN", round(np.sum(correct) / correct.size, 2))
        self.results(n, data, "NN")

    def compareMethods(self, n):
        rfData = []
        svcData = []
        nnData = []
        somewhat = []
        for i in range(0, n):
            print("Trial #", i)
            self.trainX, self.testX, self.trainY, self.testY = train_test_split(self.dataX, self.dataY, test_size=0.75)
            rf = RandomForestClassifier(n_estimators = 1000, criterion = 'gini')
            rf.fit(self.trainX, self.trainY.ravel())
            predictions = rf.predict(self.testX)
            correct = predictions == self.testY
            rate = np.sum(correct) / correct.size
            rfData.append(rate)
            print("Percentage correct for classification in RF", round(np.sum(correct) / correct.size, 2))

            s = svm.SVC(gamma='scale', kernel='rbf')
            s.fit(self.trainX, self.trainY.ravel())
            predictions = s.predict(self.testX)
            correct = predictions == self.testY
            rate = np.sum(correct) / correct.size
            svcData.append(rate)
            print("Percentage correct for classification in SVM", round(np.sum(correct) / correct.size, 2))

            nn = MLPClassifier(hidden_layer_sizes=(100, 75, 50), activation='tanh', solver='sgd', alpha=.01, max_iter=200)
            nn.fit(self.trainX, self.trainY.ravel())
            predictions = nn.predict(self.testX)
            correct = predictions == self.testY
            rate = np.sum(correct) / correct.size
            nnData.append(rate)
            print("Percentage correct for classification in NN", round(np.sum(correct) / correct.size, 2))

            guessSomewhat = np.empty_like(self.testY)
            guessSomewhat[:] = 'Somewhat masculine'
            correctSomewhat = guessSomewhat == self.testY
            rate = np.sum(correctSomewhat) / correctSomewhat.size
            somewhat.append(rate)
            print("Percentage correct for guessing somewhat", round(np.sum(correctSomewhat) / correctSomewhat.size, 2))
        self.resultsCompare(n, rfData, svcData, nnData, somewhat)


def main():
    parser = argparse.ArgumentParser(description='ML Main')
    parser.add_argument('-alg',
                        choices=['rf', 'svm', 'nn', 'all'],
                        required=True,
                        help='Algorithm to use')
    parser.add_argument('-n',
                        required=True,
                        help='Number of trials')
    parser.add_argument('-estimators',
                        required=False,
                        help='Number of trees in RF forest')
    parser.add_argument('-criterion',
                        choices=['gini', 'entropy'],
                        required=False,
                        help='Split criterion for RF')
    parser.add_argument('-depth',
                        required=False,
                        help='Maximum depth for RF')
    parser.add_argument('-kernel',
                        choices=['poly', 'rbf', 'sigmoid'],
                        required=False,
                        help='Type of kernel')
    parser.add_argument('-gamma',
                        required=False,
                        help='Float kernel coefficient')
    parser.add_argument('-solver',
                        choices=['lbfgs', 'sgd', 'adam'],
                        required=False,
                        help='Solver for weight optimization')
    parser.add_argument('-activation',
                        choices=['identity', 'logistic', 'tanh', 'relu'],
                        required=False,
                        help='Activation function for hidden layer')
    parser.add_argument('-alpha',
                        required=False,
                        help='Alpha for NN')
    parser.add_argument('-iterations',
                        required=False,
                        help='The number of iterations in NN')
    parser.add_argument('-seed',
                        default=None,
                        required=False,
                        help='Optional random number seed')

    args = parser.parse_args()
    n = int(args.n)
    m = Masculinity("raw-responses.csv")

    if args.seed:
        np.random.seed(int(args.seed))
    if args.alg == "rf":
        criterion = 'gini'
        estimators = 1000
        depth = None
        if(args.criterion):
            criterion = args.criterion
        if(args.estimators):
            estimators = int(args.estimators)
        if(args.depth):
            depth = int(args.depth)
        m.rf(n, estimators, criterion, depth)
    if args.alg == "svm":
        kernel = 'rbf'
        gamma = 'scale'
        if(args.kernel):
            kernel = args.kernel
        if(args.gamma):
            gamma = float(args.gamma)
        m.svm(n, kernel, gamma)
    if args.alg == "nn":
        solver = 'sgd'
        activation = 'tanh'
        alpha = 0.01
        iterations = 200
        if(args.solver):
            solver = args.solver
        if(args.activation):
            activation = args.activation
        if(args.alpha):
            alpha = float(args.alpha)
        if(args.iterations):
            iterations = int(args.iterations)
        m.nn(n, activation, solver, alpha, iterations)
    if args.alg == "all":
        m.compareMethods(n)


if __name__ == '__main__':
    main()