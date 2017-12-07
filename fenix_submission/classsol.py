# Grupo 081; Pedro Caetano 56564; Francisco Henriques 75278;

import numpy as np
from sklearn import neighbors, datasets, tree, linear_model

from sklearn.externals import joblib
import timeit

from sklearn.model_selection import cross_val_score

#
from sklearn.naive_bayes import MultinomialNB

import pandas as pd

from string import ascii_lowercase
from collections import Counter

ascii_vowels_lowercase = 'aeiou'
Y_LOWERCASE = 'y'
ascii_consonants_lowercase = ''.join(set(ascii_lowercase) - set(ascii_vowels_lowercase))


class Features:

    def is_asccii(x):
        return set(x).issubset(ascii_lowercase)

    def number_of_vowels(x):
#        counter = Counter(x); n = sum(counter[k] for k,v in counter.items() if k in ascii_vowels_lowercase)
        s = pd.Series(Counter(x))
        n = s.get(list(ascii_vowels_lowercase), pd.Series()).sum()
        if not x.startswith(Y_LOWERCASE):
            n += s.get(Y_LOWERCASE, 0)
        return n 

    def number_of_vowels_is_even(x):
        return Features.number_of_vowels(x) % 2 == 0

    def number_of_consonants(x):
        return pd.Series(Counter(x)).get(list(ascii_consonants_lowercase), pd.Series()).sum() 

    def false_characters(x):
        s = {'ç', 'k', '7', '4', 'a'}
        return any(set(x).intersection(s))
   
    def false_last_character(x):
        s = {'z', 'k', 'i', 'ã', '7', 'b', 'í', 'x', 'a', 'ú'}
        return x[-1] in s

    def has_digits(x):
        s = set(map(str, range(10)))
        return any(set(x).intersection(s))


def features(X):
    
    F = np.zeros((len(X),5))
    for x in range(0,len(X)):
        F[x,0] = len(X[x])
        F[x,1] = Features.number_of_vowels_is_even(X[x])
        F[x,2] = not Features.has_digits(X[x])
        F[x,3] = not Features.false_last_character(X[x])
        F[x,4] = not Features.false_characters(X[x])

    return F     

def mytraining(f,Y):

    #clf = MultinomialNB()
    clf=tree.DecisionTreeClassifier()

    clf.fit(f, Y)
   
    return clf
    
def mytrainingaux(f,Y,par):
    
    return clf

def myprediction(f, clf):
    Ypred = clf.predict(f)

    return Ypred

