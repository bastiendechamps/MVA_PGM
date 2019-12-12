import matplotlib.pyplot as plt
import numpy as np
import data_loader
from lda import lda_experiment
from logistic import logistic_regression_experiment
from linear import linear_regression_experiment
from qda import qda_experiment


if __name__ == "__main__":
    # Train datasets
    X_trainA, Y_trainA = data_loader.load("data/trainA")
    X_trainB, Y_trainB = data_loader.load("data/trainB")
    X_trainC, Y_trainC = data_loader.load("data/trainC")

    # Test datasets
    X_testA, Y_testA = data_loader.load("data/testA")
    X_testB, Y_testB = data_loader.load("data/testB")
    X_testC, Y_testC = data_loader.load("data/testC")

    # 2.1 : LDA
    print("{:=^30}".format("LDA"))
    lda_experiment(X_trainA, Y_trainA, X_testA, Y_testA, "A")
    lda_experiment(X_trainB, Y_trainB, X_testB, Y_testB, "B")
    lda_experiment(X_trainC, Y_trainC, X_testC, Y_testC, "C")

    # 2.2 : logistic regression
    print("{:=^30}".format("Logistic Regression"))
    logistic_regression_experiment(X_trainA, Y_trainA, X_testA, Y_testA, "A")
    logistic_regression_experiment(X_trainB, Y_trainB, X_testB, Y_testB, "B")
    logistic_regression_experiment(X_trainC, Y_trainC, X_testC, Y_testC, "C")

    # 2.3 : linear regression
    print("{:=^30}".format("Linear Regression"))
    linear_regression_experiment(X_trainA, Y_trainA, X_testA, Y_testA, "A")
    linear_regression_experiment(X_trainB, Y_trainB, X_testB, Y_testB, "B")
    linear_regression_experiment(X_trainC, Y_trainC, X_testC, Y_testC, "C")

    # 2.5 : QDA
    print("{:=^30}".format("QDA"))
    qda_experiment(X_trainA, Y_trainA, X_testA, Y_testA, "A")
    qda_experiment(X_trainB, Y_trainB, X_testB, Y_testB, "B")
    qda_experiment(X_trainC, Y_trainC, X_testC, Y_testC, "C")
