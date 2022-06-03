pip install -U scikit-learn 

pip install seaborn 


import matplotlib.pyplot as plt
from matplotlib import cm

import numpy as np
import pandas as pd

import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.img_tiles as cimgt

from IPython.display import display_markdown
import pylab as pl

from sklearn.metrics import r2_score
import math

import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std

from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d

import seaborn as sns


x = [-2.5, -2.4, -2.3, -2.2, -2.1, -2, -1.8, 
       -1.7, -1.6, -1.5, -1.42, -1.35, -1.3,
       -1.25, -1.2, -1.1, -1, -0.95, -0.9,-0.85,
       -0.8, -0.7, -0.65, -0.6, -0.55, -0.5]  


y = [0.425, 0.43, 0.426, 0.428, 0.44, 0.445,0.425, 
       0.41, 0.4, 0.385, 0.35, 0.325, 0.295, 0.275,
       0.26, 0.225, 0.2, 0.192, 0.185, 0.18, 0.175, 
       0.155, 0.15, 0.125, 0.115, 0.1]  


def temperature_fitting(x,y):

    z = np.polyfit(x, y, 3) 

    p = np.poly1d(z)

    print('The fitted polynomial is:')
    print(p) 

    yvals=p(x)

    r2 = r2_score(y, yvals)
    
    peak = np.polyder(p, 1)
    print('peak value=', peak.r)

    plot1 = plt.plot(x, y, 'o', label='original values', color='c')
    plot2 = plt.plot(x, yvals, label='fitting values', color='m' ,linewidth=2)

    plt.xlabel('Temperature (°)')
    plt.ylabel('Friction coefficient, μ')
    plt.legend(loc="best")  
    plt.title('Fitting diagram')
    plt.show()
    print('The polynomial fit R-square is:')
    return r2

temperature_fitting(x,y)

x = [0, 10, 20, 24, 40, 50]  

y = [0.2, 0.3, 0.45, 0.49, 0.46, 0.55] 

def debris_3_fitting(x,y):
   
    z = np.polyfit(x, y, 1) 

    p = np.poly1d(z)

    print('The fitted polynomial is:',p)

    yvals=p(x)

    r2 = r2_score(y, yvals)

    plot1 = plt.plot(x, y, 'v', label='original values', color='b')
    plot2 = plt.plot(x, yvals, color='m', label='fitting values', linewidth=2)
    
    plt.xlabel('% Debris')
    plt.ylabel('Friction coefficient, μ')
    plt.legend(loc="best")  
    plt.title('Fitting diagram')
    plt.show()
    print('The polynomial fit R-square is:')
    return r2

debris_3_fitting(x,y) 

x = [0, 14, 24.5, 33, 41]  

y = [0.35, 0.53, 0.59, 0.65, 0.68]  

def debris_6_fitting(x,y):

    z = np.polyfit(x, y, 1) 

    p = np.poly1d(z)

    print('The fitted polynomial is:',p)

    yvals=p(x)

    r2 = r2_score(y, yvals)

    plot1 = plt.plot(x, y, '*', label='original values', color='g')
    plot2 = plt.plot(x, yvals, color='m', label='fitting values', linewidth=2)
    
    plt.xlabel('% Debris')
    plt.ylabel('Friction coefficient, μ')
    plt.legend(loc="best")  
    plt.title('Fitting diagram')
    plt.show()
    print('The polynomial fit R-square is:')
    return r2

debris_6_fitting(x,y) 

lambdaa = 0.183 
a = 0.0153
N = 500000

s = float(input('please input s in (0,1):')) 

def value_slip_velocity(s,lambdaa,a,N):
    
    pi = math.pi
    
    W = 2*pi/lambdaa
    
    trans1 = (1-math.cos(2*pi*s))/(2*pi*(1-s)+ math.sin(2*pi*s))
    
    XC = (1/W)*math.atan(trans1)

    trans2 = math.sin(pi*s)-pi*s*math.cos(pi*s)
    
    phi = ((pi*s-0.5*math.sin(2*pi*s))*math.sin(pi*s-W*XC))/(trans2)
    
    tau = a*W*N*phi/2/1000000

    print("value of shear strain =",tau,"Mpa")
    
    slip_velocity = 0.3*tau
    
    print("slip velocity of glacier =",slip_velocity,"m/s")
    return tau

value_slip_velocity(s,lambdaa,a,N) 

def value_slip_velocity_return(s,lambdaa,a,N):
    
    pi = math.pi
    
    W = 2*pi/lambdaa
    
    trans1 = (1-math.cos(2*pi*s))/(2*pi*(1-s)+ math.sin(2*pi*s))
    
    XC = (1/W)*math.atan(trans1)

    trans2 = math.sin(pi*s)-pi*s*math.cos(pi*s)
    
    phi = ((pi*s-0.5*math.sin(2*pi*s))*math.sin(pi*s-W*XC))/(trans2)
    
    tau = a*W*N*phi/2/1000000
    
    slip_velocity = 0.3*tau
    
    return slip_velocity # return the value of slip_velocity

list1 = np.arange(0.1,1,0.1)
list2 = []
for i in list1:

    q = value_slip_velocity_return(i,lambdaa=0.183,a=0.0153,N=500000)

    list2.append(q)

m = list1 
n = list2 

print("Below is a plot of relationship between s and slip velocity.")

plot1 = plt.plot(m, n, 'o', label='velocity values, m/s', color='m')

plt.xlabel('values of s')
plt.ylabel('values of velocity')
plt.legend(loc="best") 
plt.title('velocity diagram')
plt.show()

x = np.arange(-2.5,-0.45,0.08)
y = np.arange(0.5,0.12,-0.015)
z = [0.425, 0.43, 0.426, 0.428, 0.44, 0.445,0.425, 
        0.41, 0.4, 0.385, 0.35, 0.325, 0.295, 0.275,
        0.26, 0.225, 0.2, 0.192, 0.185, 0.18, 0.175, 
        0.155, 0.15, 0.125, 0.115, 0.1] 

def fitting_multi(x,y,z):    
    x0 = np.ones(26)  
    x1 = x
    x2 = y

    X = np.column_stack((x0, x1, x2)) 
    yTest = z
    model = sm.OLS(yTest, X) 
    results = model.fit()  
    yFit = results.fittedvalues  
    print(results.summary())  
    print("\nOLS model: Y = b0 + b1*X + ... + bm*Xm")
    print('Parameters: ', results.params)     

    prstd, ivLow, ivUp = wls_prediction_std(results) 
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(x1, yTest, 'o', label="data")  
   
    ax.plot(x1, yFit, 'r-', label="OLS")  
    ax.plot(x1, ivUp, '--',color='orange', label="ConfInt") 
    ax.plot(x1, ivLow, '--',color='orange') 
    ax.legend(loc='best') 
    plt.xlabel('x1')
    plt.ylabel('y')
    plt.show()
    
    
    F = results.params
    f1 = F[0]
    f2 = F[1]
    f3 = F[2]
    
    
    fig = plt.figure()  
    ax3 = plt.axes(projection='3d')

    X, Y = np.meshgrid(x1, x2)
    Z = f1 + f2*X + f3*Y

    ax3.plot_surface(X,Y,Z,cmap='rainbow')
    #ax3.contour(X,Y,Z, zdim='z',offset=-2，cmap='rainbow) 
    
    ax3.scatter(x1,x2,yTest,c='skyblue')
    
    plt.xlabel('Temperature (°)')
    plt.ylabel('Debris, %')
    ax3.set_zlabel('Friction Coefficient')
    plt.title('Fitting diagram')
    plt.show()
    
    return results.params

fitting_multi(x,y,z)

x = np.arange(-2.5,-0.45,0.08)
y = np.arange(0.5,0.12,-0.015)
[X, Y] = np.meshgrid(x, y)

x_size = x.size
y_size = y.size
z = np.zeros((y_size, x_size))
# print(y.size)
for i in range(x_size):
    for j in range(y_size):
        z[j][i] = -0.00371976 + (-0.18800622 * x[i]) + (0.03513492 * y[j])

def neural_network(x,y,z,X,Y,x_size, y_size):
    hidesize = 7  # 隐层数量
    W1x = np.random.random((hidesize, 1)) 
    W1y = np.random.random((hidesize, 1))  
    B1 = np.random.random((hidesize, 1)) 
    W2 = np.random.random((1, hidesize))  
    B2 = np.random.random((1, 1)) 
    threshold = 0.007 
    max_steps = 20  


    def sigmoid(x_):  
        y_ = 1 / (1 + math.exp(-x_))
        return y_


    E = np.zeros((max_steps, 1))  
    Z = np.zeros((x_size, y_size)) 
    for k in range(max_steps):
        temp = 0
        for i in range(x_size):
            for j in range(y_size):
                hide_in = np.dot(x[i], W1x) + np.dot(y[j], W1y) - B1  
                # print(x[i])
                hide_out = np.zeros((hidesize, 1))
                for m in range(hidesize):
    
                    hide_out[m] = sigmoid(hide_in[m]) 
               
                    z_out = np.dot(W2, hide_out) - B2

                Z[j][i] = z_out

                e = z_out - z[j][i] 
    
                dB2 = -1 * threshold * e
                dW2 = e * threshold * np.transpose(hide_out)
                dB1 = np.zeros((hidesize, 1))
                for m in range(hidesize):
                    dB1[m] = np.dot(np.dot(W2[0][m], sigmoid(hide_in[m])), (1 - sigmoid(hide_in[m])) * (-1) * e * threshold)
                dW1x = np.zeros((hidesize, 1))
                dW1y = np.zeros((hidesize, 1))

                for m in range(hidesize):
                    dW1y[m] = np.dot(np.dot(W2[0][m], sigmoid(hide_in[m])), (1 - sigmoid(hide_in[m])) * y[j] * e * threshold)
                W1y = W1y - dW1y
                for m in range(hidesize):
                    dW1x[m] = np.dot(np.dot(W2[0][m], sigmoid(hide_in[m])), (1 - sigmoid(hide_in[m])) * x[i] * e * threshold)
                W1x = W1x - dW1x
                B1 = B1 - dB1
                W2 = W2 - dW2
                B2 = B2 - dB2
                temp = temp + abs(e)

        E[k] = temp

        if k % 2 == 0:
            print(k)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_title("z=-0.00371976 + -0.18800622*x + 0.03513492*y")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")


    ax.plot_surface(X, Y, z, cmap='rainbow')
    plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_title("fitting z=-0.00371976 + -0.18800622*x + 0.03513492*y")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    print('e:')
    print(E)
    ax.plot_surface(X, Y, Z, cmap='rainbow')
    plt.show()


    return k

neural_network(x,y,z,X,Y,x_size, y_size)


X1 = np.arange(-2.5,-0.45,0.08)
X2 = np.arange(0.5,0.12,-0.015)
Y = [0.425, 0.43, 0.426, 0.428, 0.44, 0.445,0.425, 
       0.41, 0.4, 0.385, 0.35, 0.325, 0.295, 0.275,
       0.26, 0.225, 0.2, 0.192, 0.185, 0.18, 0.175, 
       0.155, 0.15, 0.125, 0.115, 0.1]

df = pd.read_table('data.txt',sep=',',header=None,names=['X1','X2','Y'])

print(df) 

df.describe() 

X = (df[['X1', 'X2']] - df[['X1', 'X2']].mean()) / df[['X1', 'X2']].std()
Y = (df['Y'] - df['Y'].mean()) / df['Y'].std()
X = np.array(X)
Y = np.array(Y)

figure = plt.figure(figsize=(8,8))
ax = plt.subplot(111, projection='3d')  
ax.scatter(X1, X2, Y, c='g') 
ax.set_zlabel('Z')  
ax.set_ylabel('Y')
ax.set_xlabel('X')
plt.show()

sns.set(style="ticks", color_codes=True)
sns.pairplot(df, x_vars=['X1', 'X2'], y_vars='Y', height=4, aspect=0.8)
plt.show()

graph = sns.pairplot(df, kind="reg")

def return_Y_estimate(theta_now, x):

    theta_now = theta_now.reshape(-1, 1)
    _Y_estimate = np.dot(x, theta_now)

    return _Y_estimate


def return_dJ(theta_now, x, y_true):
    y_estimate = return_Y_estimate(theta_now, x)

    size = x.shape[0]

    _num_of_features = x.shape[1]

    dJ = np.zeros([_num_of_features, 1])
    for i in range(_num_of_features):
        dJ[i, 0] = 2 * np.dot((y_estimate - y_true).T, x[:, i]) / size
    return dJ

def return_J(theta_now, x, y_true):

    length = x.shape[0]
    temp = y_true - np.dot(x, theta_now)
    J = np.dot(temp.T, temp) / length

    return J


def gradient_descent(x,y,error,Learning_rate=0.01,ER=1e-10,MAX_LOOP=1e5):

    _num_of_samples = x.shape[0]

    X_0 = np.ones([_num_of_samples, 1])
    new_x = np.column_stack((X_0, x))

    new_y = y.reshape(-1, 1)

    _num_of_features = new_x.shape[1]

    theta = np.zeros([_num_of_features, 1]) * 0.3
    flag = 0 
    last_J = 0 
    ct = 0  

    while flag == 0 and ct < MAX_LOOP:
        last_theta = theta

        gradient = return_dJ(theta, new_x, new_y)
        theta = theta - Learning_rate * gradient
        er = abs(return_J(last_theta, new_x, new_y) - return_J(theta, new_x, new_y))
        error.append(er[0][0])
 

        if er < ER:
            flag = 1

        ct += 1
    return theta


global error
error=[]


theta = gradient_descent(X, Y,error)
print(theta)


x1,x2=np.mgrid[1:1000:1,1:1000:1]
y_pred=np.array(x1)*theta[1]+np.array(x2)*theta[2]+theta[0]

x1=(x1-x1.mean())/x1.std()
x2=(x2-x2.mean())/x2.std()
y_pred=(y_pred-y_pred.mean())/y_pred.std()

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(X1,X2,Y,color='b')
ax.scatter(x1,x2,y_pred,color='g')

ax.set_zlabel('Z', fontdict={'size': 15})
ax.set_ylabel('Y', fontdict={'size': 15})
ax.set_xlabel('X', fontdict={'size': 15})
plt.show()

plt.plot(np.arange(1,262,1),error)
plt.xlabel('number of iterations')
plt.ylabel('error')
plt.title('error and number of iterations')


X1=(df['X1']-df['X1'].mean())/df['X1'].std()
X2=(df['X2']-df['X2'].mean())/df['X2'].std()
Y=(df['Y']-df['Y'].mean())/df['Y'].std()


x1,x2=np.mgrid[1:1000:1,1:1000:1]
y_pred=np.array(x1)*theta[1]+np.array(x2)*theta[2]+theta[0]

x1=(x1-x1.mean())/x1.std()
x2=(x2-x2.mean())/x2.std()
y_pred=(y_pred-y_pred.mean())/y_pred.std()
import matplotlib.pyplot as plt

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(x1,x2,y_pred,color='b')
ax.scatter(X1,X2,Y,color='r')

ax.set_zlabel('Z', fontdict={'size': 10})
ax.set_ylabel('Y', fontdict={'size': 10})
ax.set_xlabel('X', fontdict={'size': 10})
plt.show()
