# import useful packages
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

# read the data from publications

def temperature_fitting(x,y):

    # Perform polynomial fitting (select 3rd degree polynomial fitting here)
    z = np.polyfit(x, y, 3) 

    # Get the fitted polynomial
    p = np.poly1d(z)

    print('The fitted polynomial is:')
    print(p)  # Print the fitted polynomial on the screen

    # Calculate the fitted y value
    yvals=p(x)

    # Calculate the R-square after fitting to detect the fitting effect
    r2 = r2_score(y, yvals)
    
    
    # Calculate the extreme points of the fitted polynomial
    peak = np.polyder(p, 1)
    print('peak value=', peak.r)

    # Draw plots to compare and analyze
    plot1 = plt.plot(x, y, 'o', label='original values', color='c')
    plot2 = plt.plot(x, yvals, label='fitting values', color='m' ,linewidth=2)

    plt.xlabel('Temperature (°)')
    plt.ylabel('Friction coefficient, μ')
    plt.legend(loc="best")  
    plt.title('Fig 1. Fitting diagram')
    plt.show()
    print('The polynomial fit R-square is:')
    return r2
# temperature_fitting() 
# Fitting the relationship between friction coefficient and bedrock temperature by function np.polyfit

# read the data from publications

def debris_3_fitting(x,y):
   
    # Perform polynomial fitting (select 1st degree polynomial fitting here)
    z = np.polyfit(x, y, 1) 

    # Get the fitted polynomial
    p = np.poly1d(z)

    print('The fitted polynomial is:',p)
    # Print the fitted polynomial on the screen

    # Calculate the fitted y value
    yvals=p(x)

    # Calculate the R-square after fitting to detect the fitting effect
    r2 = r2_score(y, yvals)

    # Draw plots to compare and analyze
    plot1 = plt.plot(x, y, 'v', label='original values', color='b')
    plot2 = plt.plot(x, yvals, color='m', label='fitting values', linewidth=2)
    
    plt.xlabel('% Debris')
    plt.ylabel('Friction coefficient, μ')
    plt.legend(loc="best")  
    plt.title('Fig 2. Fitting diagram')
    plt.show()
    print('The polynomial fit R-square is:')
    return r2
# debris_3_fitting()
# Fitting the relationship between friction coefficient and debris at -3 Celsius by function np.polyfit

# read the data from publications

def debris_6_fitting(x,y):

    # Perform polynomial fitting (select 1st degree polynomial fitting here)
    z = np.polyfit(x, y, 1) 

    # Get the fitted polynomial
    p = np.poly1d(z)

    print('The fitted polynomial is:',p)
    # Print the fitted polynomial on the screen

    # Calculate the fitted y value
    yvals=p(x)

    # Calculate the R-square after fitting to detect the fitting effect
    r2 = r2_score(y, yvals)

    # Draw plots to compare and analyze
    plot1 = plt.plot(x, y, '*', label='original values', color='g')
    plot2 = plt.plot(x, yvals, color='m', label='fitting values', linewidth=2)
    
    plt.xlabel('% Debris')
    plt.ylabel('Friction coefficient, μ')
    plt.legend(loc="best")  
    plt.title('Fig 3. Fitting diagram')
    plt.show()
    print('The polynomial fit R-square is:')
    return r2
# debris_6_fitting()
# Fitting the relationship between friction coefficient and debris at -6 Celsius by function np.polyfit


def value_slip_velocity(s,lambdaa,a,N):
    
    pi = math.pi
    
    W = 2*pi/lambdaa
    
    trans1 = (1-math.cos(2*pi*s))/(2*pi*(1-s)+ math.sin(2*pi*s))
    
    XC = (1/W)*math.atan(trans1)

    trans2 = math.sin(pi*s)-pi*s*math.cos(pi*s)
    
    phi = ((pi*s-0.5*math.sin(2*pi*s))*math.sin(pi*s-W*XC))/(trans2)
    
    tau = a*W*N*phi/2/1000000
    # The equation comes from Lucas K. Zoet and Neal R. Iverson, 2017.
    # For a sinusoid of angular frequency (i.e. wavenumber), ω = 2π/λ, and in a direction, x, 
    # parallel to the regional bed slope, the bed shear stress (drag force per unit bed area), τ, is a*W*N*phi

    print("value of shear strain =",tau,"Mpa")
    
    slip_velocity = 0.3*tau
    
    print("slip velocity of glacier =",slip_velocity,"m/s")
    return tau
# value_slip_velocity(s) # obtain the value of shear strain and slip velocity of glacier we want


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




def fitting_multi(x,y,z):    
    x0 = np.ones(26)  # Intercept column x0=[1,...1]
    x1 = x
    x2 = y

    X = np.column_stack((x0, x1, x2)) 
    yTest = z
    # Multiple Linear Regression: Least Squares (OLS)
    model = sm.OLS(yTest, X)  # Build the OLS model: Y = b0 + b1*X + ... + bm*Xm + e
    results = model.fit()  # Returns model fitting results
    yFit = results.fittedvalues  # the y-value for the model fit
    print(results.summary())  # A summary of the output regression analysis
    print("\nOLS model: Y = b0 + b1*X + ... + bm*Xm")
    print('Parameters: ', results.params)  # Output: Coefficients of the fitted model

    # Plotting: raw data points, fitted curves, confidence intervals
    prstd, ivLow, ivUp = wls_prediction_std(results) # Returns the standard deviation and confidence intervals
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(x1, yTest, 'o', label="data")  # Experimental data (raw data + error)
   
    ax.plot(x1, yFit, 'r-', label="OLS")  # fit data
    ax.plot(x1, ivUp, '--',color='orange', label="ConfInt")  # Confidence interval upper bound
    ax.plot(x1, ivLow, '--',color='orange')  # Confidence interval lower bound
    ax.legend(loc='best')  # show legend
    plt.xlabel('x1')
    plt.ylabel('y')
    plt.title('Fig 5. Fitting diagram')
    plt.show()
    
    
    F = results.params
    f1 = F[0]
    f2 = F[1]
    f3 = F[2]
    
    
    fig = plt.figure()  # Define a new 3D axis
    ax3 = plt.axes(projection='3d')

    # Define 3D data
    X, Y = np.meshgrid(x1, x2)
    Z = f1 + f2*X + f3*Y
    # Plot

    ax3.plot_surface(X,Y,Z,cmap='rainbow')
    #ax3.contour(X,Y,Z, zdim='z',offset=-2，cmap='rainbow)   # Contour map, to set the offset, the minimum value of Z
    
    ax3.scatter(x1,x2,yTest,c='skyblue')
    
    plt.xlabel('Temperature (°)')
    plt.ylabel('Debris, %')
    ax3.set_zlabel('Friction Coefficient')
    plt.title('Fig 6. Fitting diagram')
    plt.show()
    
    return results.params

# ##function approximation f(x)=sin(x)
# ## The activation function uses sigmoid

def neural_network(x,y,z,X,Y,x_size, y_size):
    hidesize = 7  # number of hidden layers
    W1x = np.random.random((hidesize, 1))  # The weight between the input layer and the hidden layer
    W1y = np.random.random((hidesize, 1))  # The weight between the input layer and the hidden layer
    B1 = np.random.random((hidesize, 1))  # Threshold of hidden layer neurons
    W2 = np.random.random((1, hidesize))  # The weight between the hidden layer and the output layer
    B2 = np.random.random((1, 1))  # Threshold for output layer neurons
    threshold = 0.007  # Threshold
    max_steps = 20  # The maximum number of iterations, after which it will exit


    def sigmoid(x_):  # Here x_ and y_ are in the function and do not need to be changed
        y_ = 1 / (1 + math.exp(-x_))
        return y_


    E = np.zeros((max_steps, 1))  # Error as a function of number of iterations
    Z = np.zeros((x_size, y_size))  # The output of the model
    for k in range(max_steps):
        temp = 0
        for i in range(x_size):
            for j in range(y_size):
                hide_in = np.dot(x[i], W1x) + np.dot(y[j], W1y) - B1  # Hidden layer input data
                # print(x[i])
                hide_out = np.zeros((hidesize, 1))  # The output data of the hidden layer
                for m in range(hidesize):
                    # print ("The value of the {}th is {}".format(j,hide_in[j]))
                    # print(j,sigmoid(j))
                    hide_out[m] = sigmoid(hide_in[m])  # Calculate hide_out
                    # print("The value of the {}th is {}".format(j, hide_out[j]))

                    # print(hide_out[3])
                    z_out = np.dot(W2, hide_out) - B2  # model output
                    
                Z[j][i] = z_out
                # print(i,Y[i])

                e = z_out - z[j][i]  # Model output minus actual results. get error

                # feedback, modify parameters
                dB2 = -1 * threshold * e
                dW2 = e * threshold * np.transpose(hide_out)
                dB1 = np.zeros((hidesize, 1))
                for m in range(hidesize):
                    dB1[m] = np.dot(np.dot(W2[0][m], sigmoid(hide_in[m])), (1 - sigmoid(hide_in[m])) * (-1) * e * threshold)
                    # np.dot((sigmoid(hide_in[j])), (1 - sigmoid(hide_in[j]))) is the derivative of sigmoid(hide_in[j])
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
    # new a figure and set it into 3d
    fig = plt.figure()
    # set figure information
    ax = plt.axes(projection='3d')
    ax.set_title("Fig 7. z=-0.00371976 + -0.18800622*x + 0.03513492*y")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    # draw the figure, the color is r = read

    ax.plot_surface(X, Y, z, cmap='rainbow')
    # The error function graph can directly subtract the two function values Y and y above.
    plt.figure()
    ax = plt.axes(projection='3d')
    # set figure information
    ax.set_title("Fig 8. fitting z=-0.00371976 + -0.18800622*x + 0.03513492*y")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    # print(x)
    # print(z)
    # print(Z)
    # draw the figure, the color is r = read
    print('e:')
    print(E)
    ax.plot_surface(X, Y, Z, cmap='rainbow')
    plt.show()
    # The error function graph can directly subtract the two function values Y and y above.

    return k


#Add a column of 1 to the x vector, that is, set X_0 = 1
#Enter the current theta and x, and add a column to x
#Output the result of multiplying the theta vector by the x matrix
def return_Y_estimate(theta_now, x):
    # Make sure theta_now is a column vector
    theta_now = theta_now.reshape(-1, 1)
    _Y_estimate = np.dot(x, theta_now)

    return _Y_estimate

# Find the gradient of the current theta
# Enter the current theta, the independent variable x, the real dependent variable value y_true, and return the gradient of the current theta
def return_dJ(theta_now, x, y_true):
    y_estimate = return_Y_estimate(theta_now, x)
    # A total of size group data
    size = x.shape[0]
    # Number of thetas to solve
    _num_of_features = x.shape[1]
    # Construct
    dJ = np.zeros([_num_of_features, 1])
    for i in range(_num_of_features):
        dJ[i, 0] = 2 * np.dot((y_estimate - y_true).T, x[:, i]) / size
    return dJ

# Calculate the value of the loss function J
# Enter the current theta value, independent variable, dependent variable, and return the value to calculate J
def return_J(theta_now, x, y_true):
    # A total of N groups of data
    length = x.shape[0]
    temp = y_true - np.dot(x, theta_now)
    J = np.dot(temp.T, temp) / length

    return J

# Gradient descent method to solve linear regression function
# Enter a row of x for a set of data, y is a column vector Learning rate Learning_rate defaults to 0.3 Error ER defaults to 1e-8 The default maximum number of iterations MAX_LOOP is 1e4
# Return the model parameter theta that has been iterated and converged
def gradient_descent(x,y,error,Learning_rate=0.01,ER=1e-10,MAX_LOOP=1e5):
    # The number of samples is
    _num_of_samples = x.shape[0]
    # Splice all 1 column on the leftmost side of x
    X_0 = np.ones([_num_of_samples, 1])
    new_x = np.column_stack((X_0, x))
    # make sure y is a column vector
    new_y = y.reshape(-1, 1)
    # The number of unknown elements to be solved is
    _num_of_features = new_x.shape[1]
    # initialize theta vector
    theta = np.zeros([_num_of_features, 1]) * 0.3
    flag = 0  # Define the jump out flag
    last_J = 0  # Used to store the value of the last Lose Function
    ct = 0  # used to calculate the number of iterations

    while flag == 0 and ct < MAX_LOOP:
        last_theta = theta
        # update theta
        gradient = return_dJ(theta, new_x, new_y)
        theta = theta - Learning_rate * gradient
        er = abs(return_J(last_theta, new_x, new_y) - return_J(theta, new_x, new_y))
        error.append(er[0][0])
 
        # When the error reaches the threshold, the jump flag will be refreshed
        if er < ER:
            flag = 1
        # Stacking Iterations
        ct += 1
    return theta
# Call the multiple regression function model
