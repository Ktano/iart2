# Grupo 081; Pedro Caetano 56564; Francisco Henriques 75278;

import numpy as np
from sklearn import datasets, tree, linear_model
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import cross_val_score
import timeit

def mytraining(X,Y):

    reg= KernelRidge(kernel='rbf', gamma=0.1,alpha=0.001)
    #reg=tree.DecisionTreeRegressor(min_samples_split=2)
    reg.fit(X,Y)
    return reg
    
def mytrainingaux(X,Y,par):

    reg= KernelRidge(kernel='rbf', gamma=par,alpha=0.1)
    reg.fit(X,Y)
                
    return reg

def myprediction(X,reg):

    Ypred = reg.predict(X)

    return Ypred

