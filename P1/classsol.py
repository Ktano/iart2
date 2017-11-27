import numpy as np
from sklearn import neighbors, datasets, tree, linear_model

from sklearn.externals import joblib
import timeit

from sklearn.model_selection import cross_val_score

def features(X):
    
    F = np.zeros((len(X),5))
    for x in range(0,len(X)):
        F[x,0] = len(X[x])%2
        F[x,1] = 1#isVowel(X[x][0]) #starts with a Vowel
        F[x,2] = 1#pairVowels(X[x]) 
        F[x,3] = 1#isVowel(X[x][-1])
        F[x,4] = NoA(X[x])

    return F     

def mytraining(f,Y):
    
    clf=tree.DecisionTreeClassifier()
    clf.fit(f,Y)
    return clf
    
def mytrainingaux(f,Y,par):
    clf=tree.DecisionTreeClassifier()
    clf.fit(f,Y)
    return clf

def myprediction(f, clf):
    Ypred = clf.predict(f)

    return Ypred

def pairVowels(s):
    tot=0
    for c in s:
        if isVowel(c):
            tot+=1
    return (tot%2)==0

def isVowel(char):
    return char.lower() in ['a','e','i','o','u']

def repeatedLetters(s):
    ls=[]
    for c in s:
        if c in ls:
            return True
        else:
            ls.append(c)
    return False

def uniqueLetters(s):
    ls=[]
    tot=0
    for c in s:
        if c not in ls:
            tot+=1
            ls.append(c)
    return tot

def NoA(s):
    for c in s:
        if c=='a':
            return False
    return True