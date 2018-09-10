import numpy as np
import matplotlib.pyplot as plt

'''X=np.array([[147,150,153,158,163,165,168,170,173,175,178,180,183]]).T
y=np.array([[49,50,51,54,58,59,60,62,63,64,66,67,68]]).T


one=np.ones((X.shape[0],1))
Xbar=np.concatenate((one,X),axis=1)

A=np.dot(Xbar.T,Xbar)
b=np.dot(Xbar.T,y)
w=np.dot(np.linalg.pinv(A),b)

print('w = ',w)

w_0=w[0][0]
w_1=w[1][0]

x0=np.linspace(145,185,2)
y0=w_0+w_1*x0

plt.plot(X,y,'ro')
plt.plot(x0,y0)
plt.axis([140,190,45,75])
plt.xlabel('Height (cm)')
plt.ylabel('Weight (kg)')
plt.show()

y1=w_1*155+w_0
y2=w_1*160+w_0

print( u'Predict weight of person with height 155 cm: %.2f (kg), real number: 52 (kg)'  %(y1) )
print( u'Predict weight of person with height 160 cm: %.2f (kg), real number: 56 (kg)'  %(y2) )

from sklearn import datasets, linear_model

regr=linear_model.LinearRegression(fit_intercept=False)
regr.fit(Xbar,y)

print( 'Solution found by scikit-learn  : ', regr.coef_ )
print( 'Solution found by TVH : ', w.T)'''

print("***************Exercise 1 quadratic regression ***************") 

X=np.array([[-3,-2,-1,0,1,2,3]]).T
y=np.array([[7.5,3,0.5,1,3,6,14]]).T



one=np.ones((X.shape[0],1))
Xbar=np.concatenate((one,X,X**2),axis=1)

A=np.dot(Xbar.T,Xbar)
b=np.dot(Xbar.T,y)
w=np.dot(np.linalg.pinv(A),b)

print('w = ',w)

w_0=w[0][0]
w_1=w[1][0]
w_2=w[2][0]

x0=np.linspace(-5,5,100)
y0=w_0+w_1*x0+w_2*x0**2

plt.plot(X,y,'ro')
plt.plot(x0,y0)
plt.axis([-5,5,-5,20])
plt.show()

from sklearn import datasets, linear_model

regr=linear_model.LinearRegression(fit_intercept=False)
regr.fit(Xbar,y)

print( 'Solution found by scikit-learn  : ', regr.coef_ )
print( 'Solution found by TVH : ', w.T)
