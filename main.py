import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numpy import linspace
from numpy.lib import dstack
from scipy.stats import multivariate_normal

class Expectation_maximization:
    def __init__(self, datapath, x, y, clusters):
        self.clusters = clusters
        data = pd.read_csv(datapath)
        data = data.dropna()
        self.xlabel = x
        self.ylabel = y
        self.x = list(data[x])
        self.y = list(data[y])
        self.datamatrix = np.vstack((self.x,self.y)).T
        self.LL = [0]
        
    def visualize(self, iteration=""):
        plt.figure()
        plt.title('\nIteration ' + str(iteration) + ': ')
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.scatter(self.x, self.y, color='green', alpha=0.2)
        plt.scatter(self.means[:, 0], self.means[:, 1], color='red')
        X, Y = np.meshgrid(np.linspace(min(self.x)-1, max(self.x)+1, 500), np.linspace(min(self.y)-1, max(self.y)+1, 500))
        pos = np.dstack((X, Y))
        for i in range(self.clusters):
            Z = multivariate_normal.pdf(pos, self.means[i, :], self.variances[i, :, :])
            plt.contour(X, Y, Z, extend='both')
        plt.show()


    def initialize(self):
        np.random.seed(1024)
        self.means = np.zeros((self.clusters, 2))
        self.variances = np.zeros((self.clusters, 2, 2))
        for i in range(self.clusters):
            self.means[i] = np.random.normal(size=2)
            self.variances[i] = np.eye(2)
        self.weights = np.ones((self.clusters, 1)) / self.clusters
        self.r = np.zeros((self.clusters, len(self.x)))

        
    def expect(self):
        for i in range(self.clusters):
            self.r[i] = self.weights[i] * multivariate_normal.pdf(self.datamatrix, self.means[i], self.variances[i], allow_singular=True)
        self.r = self.r / (np.sum(self.r, axis=0) + 1e-6)
        
    def maximize(self):
        self.weights = np.sum(self.r, axis=1) / len(self.x)
        for i in range(self.clusters):
            self.means[i] = np.sum(self.r[i] * self.datamatrix.T, axis=1) / np.sum(self.r[i])
            difference = self.datamatrix - self.means[i].T
            self.variances[i] = np.dot((self.r[i] * difference.T), difference) / np.sum(self.r[i])
        gauss = 0
        for i in range(self.clusters):
            gauss += self.weights[i] * multivariate_normal.pdf(self.datamatrix, self.means[i, :], self.variances[i, :, :], allow_singular=True)
        self.LL.append(-np.sum(np.log(gauss)))
        
    def run(self, iterations=-1):
        self.initialize()
        self.visualize("base")
        if iterations == -1:
            for i in range(10000):
                self.expect()
                self.maximize()
                if i == 0:
                    continue
                if np.absolute(self.LL[i-1] - self.LL[i]) < 1e-100:
                    self.visualize(i)
                    break
        else:
            for i in range(iterations):
                self.expect()
                self.maximize()
                self.visualize(i+1)
            
irisclassification = Expectation_maximization("./8-dataset.csv", x="Petal.Length", y="Petal.Width", clusters=3)
irisclassification.run()

lifeclassification = Expectation_maximization("./Life Expectancy Data.csv", x="Alcohol", y=" BMI ", clusters=2)
lifeclassification.run()

lifeclassification.run(5)