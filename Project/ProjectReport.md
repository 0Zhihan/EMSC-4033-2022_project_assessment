# EMSC4033 Project Report

## Exploring the influence of different factors on glacier slip

## Instructions

The kinematic status of the glacier on the bedrock are affected by many conditions, the most important of which is the coefficient of friction between the glacier and the bedrock. The coefficient of friction is also constrained by various physical conditions, such as the temperature at the base of the glacier. There are some ways in which we can obtain data on the friction coefficient and the physical conditions that affect the coefficient of friction. This project **aims to** develop a way of thinking about processing collected data with severals methods to explore the relationship between friction coefficient and several different physical conditions and to analyse how these physical conditions work.

Further, this developed way of thinking can be more **generalized** and used in other fields of research. In other words, the methods of processing data established in this project can also be applied to other problems to establish mathematical relationships between different variables. This project can help some people without programming foundation to process data more easily.

The focus of this project is to find and develop some programming methods and code packages suitable for these data to build models to analyze the data after these data is collected. The data I have obtained are the friction coefficients under conditions of different temperatures of glacier base and of  different amounts of debris glacier carrying from *L. K. Zoet et al.2013*.  The **core idea** is to fit the mathematical relationship between these conditions and the friction coefficient. At first, I fit temperature, 3 percent and 6 percent debris and the coefficient of friction, respectively. Then I plotted the theoretical value derived from the formula of glacier sliding speed from the paper. Then I fit the multivariate relationship between the friction coefficient and temperature and debris content with the OLS model. Then I use a neural network to fit the relationship between the three variables and plot it. And compare this graph with the graph fitted by the OLS model. Finally, I solve the linear regression function with gradient descent with three variables, and get the model parameters theta and plot it. 

In `my_functions.py`, data x represents temperature, data y represents percentage of debris and data z represents friction coefficient. And these functions are well explained. Functions I wrote and their corresponding effects are shown in the table below.

|Function Name | Corresponding Effects | Package and Function | 
|  --- |    --- |             --- | 
| temperature_fitting |Fit polynomial about temperature and friction coeffcient and plot|np.polyfit,  np.polyld |
| debris_3_fitting | Fit polynomial about debris of 3% and friction coeffcient and plot| np.polyfit,  np.polyld | 
| debris_6_fitting | Fit polynomial about debris of 6% and friction coeffcient and plot| np.polyfit,  np.polyld |
| value_slip_velocity  |Obtain the value of slip velocity of glacier we want when input s |math |
| value_slip_velocity_return  |Return the value of slip velocity of glacier we want when input s |math |
| fitting_multi  |Multiple Linear Regression using Least Squares (OLS Model) |sm.OLS, model.fit, results.fittedvalues|
| neural_network  |Fitting data using a neural network and testing results from the OLS model |sigmoid, np.zeros, np.random.random |
| gradient_descent  |Returns the iterated and converged model parameters theta |np.ones, np.column_stack, np.zeros |

In `Run_project.ipynb`, Figure 1, Figure 2 and Figure 3 are diagrams of unary polynomial. Figure 4 is a diagram of theoretical values of the sliding velocities at different contact surface ratios between the glacier and the bedrock. Figure 5 is a diagram of the relationship between the friction coefficient and the first variable fitted by the OLS model. Figure 6 is a 3d diagram of the relationship between the three variables fitted by the OLS model. Figure 7 is a 3d diagram of the relationship between the three variables. Figure 8 is a 3d diagram of the relationship between the three variables fitted by neural network. Figures 9 and 10 are two-dimensional and three-dimensional scatter plots of three variables, respectively. Figure 11 shows the fitted plan and scatter plot. Figure 12 is the plot of error and number of iterations. Figure 13 is the fitted plot.


## List of dependencies

**dependencies:**

- matplotlib
- numpy
- pandas
- cartopy
- IPython
- pylab
- math
- statsmodels
- mpl_toolkits
- pip:
  - sklearn
  - seaborn


## Describe testing

In `test_function.py`, I wrote eight test functions to test the functions in `my_functions.py` and execute the test in `Run_test.ipynb`. Test functions test the results of some key steps in the functions. Test functions I wrote and their corresponding effects are shown in the table below.

|Function Name | Corresponding Effects |    
|  --- |    --- | 
| test_temperature_fitting |test the data type of the R-square derived from the corresponding function |
| test_debris_3_fitting | test the data type of the R-square derived from the corresponding function | 
| test_debris_6_fitting | test the data type of the R-square derived from the corresponding function |
| test_ value_slip_velocity  |test the data type of the tau derived from the corresponding function |
| test_value_slip_velocity_return  |test the data type of the slip velocity derived from the corresponding function |
| test_fitting_multi  |test the data type of the results params derived from the corresponding function |
| test_neural_network  |test the value of the k derived from the corresponding function |
| test_gradient_descent  |test the data type of the theta derived from the corresponding function |


## Limitations/Future Improvements

The biggest limitation of this model is that it only fits a bivariate first-order polynomial, and does not involve the fitting of a multivariate higher-order relationship. In this case, the resulting data model may be simpler. At the same time, the method of neural network fitting data and  gradient descent method to solve the linear regression function can only give the model parameters and predicted values, but cannot give the specific fitted relationship. The specific one can only be obtained through the OLS model. The next step is to strengthen in this regard.
