from sklearn import neighbors, datasets
import numpy as np
from sklearn import tree
from sklearn import linear_model
from sklearn.kernel_ridge import KernelRidge


import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def kernelregress(D,xq,beta):

    kk = np.exp(-beta*np.round(np.abs(D[:,0]-xq)))
    y = np.dot(kk,D[:,1])/np.sum(kk,1)
    
    return y

#Training Set
X = np.zeros((22,1))
X[:,0] = np.arange(0,11,.5)
noisesigma = 0.2
Y = (2 + np.sin(X) + noisesigma * np.random.randn(22, 1))
#Y[[5,10,15],0] = 2 * Y[[5,10,15],0]

#Testing Set
Xp = np.zeros((110,1))
Xp[:,0] = np.arange(0,11,.1)
Yp = (2 + np.sin(Xp))


# Linear Regression
reglr = linear_model.LinearRegression()
reglr.fit(X,Y)
Ylr = reglr.predict(Xp)

# Kernel Ridge Regression
regkr = KernelRidge(kernel='rbf', gamma=0.1,alpha=0.1)
regkr.fit(X,Y)
Ykr = regkr.predict(Xp)

# Kernel Regression
Yp1 = kernelregress(np.hstack((X,Y)),Xp,10)
Yp2 = kernelregress(np.hstack((X,Y)),Xp,1)

# Decision Tree Regressor
min_samples_split = 3
regtree = tree.DecisionTreeRegressor(min_samples_split=min_samples_split)
regtree = regtree.fit(X, Y)
Ytree = regtree.predict(Xp)


plt.plot(X,Y,'go',label='true')
plt.plot(Xp,Yp1,'g',label='kerReg10')
plt.plot(Xp,Yp2,'g:',label='kerReg1')
plt.plot(Xp,Ykr,'r',label='KernRidge')
plt.plot(Xp,Ytree,'b',label='tree')
plt.plot(Xp,Ylr,'m',label='linregres')
plt.legend( loc = 3 )

plt.show()
