import numpy as np
from sklearn import neighbors, datasets, tree, linear_model
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib

import classsol_francisco as classsol

#load input data
words = []
with open("words.txt", encoding='latin1') as file:
    for line in file: 
        line = line.split(' ') #or some other preprocessing
        words.append(line) #storing everything in memory!

X = words[0]

#print('X', X)
#print('len(words)', len(words)) == 1

for test in ["wordsclass.npy", "wordsclass2.npy"]:
    print("Testing " + test)
    #load output data
    Y=np.load(test)
    
    f = classsol.features(X)    
    
    clf = classsol.mytraining(f,Y)
      
    Ypred = classsol.myprediction(f, clf)
       
    err = np.sum(Y^Ypred)/len(X) 
    print('err', err) 
    if err<.05:
        print("Erro bastante baixo. PERFECT!\n")
    elif err<.3:    
        print("Erro nos Q dentro dos limites de tolerância. OK\n")
    else:
        print("Erro nos Q acima dos limites de tolerância. FAILED\n")
