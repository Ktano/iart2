from sklearn import neighbors, datasets
import numpy as np

from sklearn import tree

from sklearn import linear_model

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

#Training Set
X = np.zeros((22,1))
X[:,0] = np.arange(0,11,.5)
noisesigma = .2
Y = np.ravel(2 + .1 * X + noisesigma * np.random.randn(22, 1))

#Testing Set
Xp = np.zeros((110,1))
Xp[:,0] = np.arange(0,11,.1)
Yp = np.ravel(2 + .1 * Xp)


# Linear Regression
reglr = linear_model.LinearRegression()
reglr.fit(X,Y)
Ylr = reglr.predict(Xp)

regridge = linear_model.RidgeCV(alphas=[0.5])
regridge.fit(X,Y)
Yridge = regridge.predict(Xp)

reglasso = linear_model.Lasso(alpha = 0.1)
reglasso.fit(X,Y)
Ylasso = reglasso.predict(Xp)


plt.plot(X,Y,'go')
plt.plot(Xp,Yp,'g',label='true')
plt.plot(Xp,Ylr,'m',label='linearregression')
plt.plot(Xp,Yridge,'b',label='ridge')
plt.plot(Xp,Ylasso,'r',label='lasso')
plt.legend( loc = 4 )

plt.show()
