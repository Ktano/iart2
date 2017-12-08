# Grupo 081; Pedro Caetano 56564; Francisco Henriques 75278;
import numpy as np
from sklearn import datasets, tree, linear_model,neighbors,svm
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import cross_val_score
import timeit

import matplotlib.pyplot as plt

def mytraining(X,Y):

    reg= KernelRidge(kernel='rbf', gamma=0.1,alpha=0.001)
    #reg=neighbors.KNeighborsRegressor(n_neighbors=1)
    reg.fit(X,Y)
    return reg
    
def mytrainingaux(X,Y,par):

    #reg=neighbors.KNeighborsRegressor(n_neighbors=par)
    reg= KernelRidge(kernel='rbf', gamma=par[0],alpha=par[1])
    reg.fit(X,Y)
                
    return reg

def myprediction(X,reg):

    Ypred = reg.predict(X)

    return Ypred


def test ():
    for ii,test in enumerate(["regress.npy", "regress2.npy"]):
        print("Testing " + test)
        for par in [1,0.1,0.01,0.001]:
            for par2 in [1,0.1,0.01,0.001]:
                
                X,Y,Xp,Yp = np.load(test)
                
                reg = mytrainingaux(X,Y,[par,par2])
                
                #Ypred = myprediction(Xp,reg)


                # if -cross_val_score( reg, X, Y, cv = 5, scoring = 'neg_mean_squared_error').mean() < tres[ii]:
                #     print("Erro dentro dos limites de tolerância. OK\n")
                # else:
                #     print("Erro acima dos limites de tolerância. FAILED\n")    
                print(str(par)+"\t"+str(par2)+"\t"+str(-cross_val_score( reg, X, Y, cv = 5, scoring = 'neg_mean_squared_error').mean()))
                # plt.figure()
                # plt.plot(Xp,Yp,'g.',label='datatesting')
                # plt.plot(X,Y,'k+',label='datatrain')
                # plt.plot(Xp,Ypred,'m',label='linregres1')
                # plt.legend( loc = 1 )
                # plt.show()


