import pytest
import numpy
from src.my_functions import *

def test_temperature_fitting(): 
    x= [-2.5, -2.4, -2.3, -2.2, -2.1, -2, -1.8, 
        -1.7, -1.6, -1.5, -1.42, -1.35, -1.3,
        -1.25, -1.2, -1.1, -1, -0.95, -0.9,-0.85,
        -0.8, -0.7, -0.65, -0.6, -0.55, -0.5]
    y=[0.425, 0.43, 0.426, 0.428, 0.44, 0.445,0.425, 
        0.41, 0.4, 0.385, 0.35, 0.325, 0.295, 0.275,
        0.26, 0.225, 0.2, 0.192, 0.185, 0.18, 0.175, 
        0.155, 0.15, 0.125, 0.115, 0.1] 
    r2 = temperature_fitting(x,y)
    type_r2 = type(r2)
    assert type_r2 == numpy.float64, " *** error is type of output is not numpy.float64 "

def test_debris_3_fitting():
    x = [0, 10, 20, 24, 40, 50]
    y = [0.2, 0.3, 0.45, 0.49, 0.46, 0.55]
    r2 = debris_3_fitting(x,y)
    type_r2 = type(r2)
    assert type_r2 == numpy.float64, " *** error is type of output is not numpy.float64 "

def test_debris_6_fitting():
    x = [0, 14, 24.5, 33, 41]
    y = [0.35, 0.53, 0.59, 0.65, 0.68]
    r2 = debris_6_fitting(x,y)
    type_r2 = type(r2)
    assert type_r2 == numpy.float64, " *** error is type of output is not numpy.float64 "

def test_value_slip_velocity():
    s=0.1
    lambdaa=0.183
    a=0.0153
    N=500000
    
    tau = value_slip_velocity(s,lambdaa,a,N)
    type_tau = type(tau)
    assert type_tau == float, " *** error is type of output is not float "

def test_value_slip_velocity_return():
    s=0.1
    lambdaa=0.183
    a=0.0153
    N=500000
    slip_velocity = value_slip_velocity_return(s,lambdaa,a,N)
    type_slip_velocity = type(slip_velocity)
    assert type_slip_velocity == float, " *** error is type of output is not float "

def test_fitting_multi():
    x=np.arange(-2.5,-0.45,0.08)
    y=np.arange(0.5,0.12,-0.015)
    z=[0.425, 0.43, 0.426, 0.428, 0.44, 0.445,0.425, 
    0.41, 0.4, 0.385, 0.35, 0.325, 0.295, 0.275,
    0.26, 0.225, 0.2, 0.192, 0.185, 0.18, 0.175, 
    0.155, 0.15, 0.125, 0.115, 0.1]
    results_params = fitting_multi(x,y,z)
    type_results_params = type(results_params)
    assert type_results_params == numpy.ndarray, " *** error is type of output is not numpy.ndarray "

def test_neural_network():
    x = np.arange(-2.5,-0.45,0.08)
    y = np.arange(0.5,0.12,-0.015)
    [X, Y] = np.meshgrid(x, y)
    x_size = x.size
    y_size = y.size
    z = np.zeros((y_size, x_size))
    
    k = neural_network(x,y,z,X,Y,x_size, y_size)
    assert k == 19, " *** error is type of output is not 19 "

def test_gradient_descent():
    global error
    error=[]
    x = np.arange(-2.5,-0.45,0.08)
    y = np.arange(0.5,0.12,-0.015)
    
    theta = gradient_descent(x,y,error)
    type_theta = type(theta)
    assert type_theta == numpy.ndarray, " *** error is type of output is not numpy.ndarray "
