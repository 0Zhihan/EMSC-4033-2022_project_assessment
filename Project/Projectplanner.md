# EMSC4033 Project Plan 

## Exploring the influence of different factors on glacier slip

## Executive summary

There is a research on kinematic status of glaciers is in my project. Thus I need to develop a data processing method to explore how the factors affecting the kinematic state of the glacier. The first step is to obtain the necessary data from the paper. There are some experiments(Zoet et al.(2013)) designed with some special devices to simulate the slip processes of glaciers to obtain data of friction coefficient as some variables like temperature of base of glaciers or percentage of debris glaciers carrying change. There are some theoretical derivations and formulas to describe the sliding law of glaciers as well.

In this case, I could fit the data with some functions to explore how these variables affect the coefficient of friction. I could obtain theoretical values of slip velocity of glaciers by formulas and compare them with actual data to verify the accuracy of the formulas as well. The specific methods used are linear fitting, neural network and multiple linear regression.

The specific outcomes will include some fitting formulas, some plots illustrating the relationships, tests of the accuracy of these formulas and calculated values and some test functions.

## Goals and Requirements

An outline of goals and requirements of this project:
- Find and import some useful packages to support my ideas.
- Access and collect data from simulation experiments about coefficient of friction and different variables.
- Fit the data with linear fitting, neural network and multiple linear regression to explore linear and nonlinear relationship between these variables and coefficient of friction.
- Test the accuracy of the fitted relation with r2 in package scikit-learn.
- Access and built some formulas of law of glaciers' slip to obtain theoretical values.
- Plot data with matplotlib and compare these plots.
- Write markdown text and docstrings to explain how these functions built and some test functions on these functions.
- Develop my github repository including reference data and package sources and updating ProjectPlanner.md

## Background and Innovation  

Some geologists have done simulation experiments of glacier slip, and obtained a series of data showing the relationship between some variables and friction coefficients under the condition of setting some basic conditions. And others have theoretically deduced the laws of glacier slip, especially in the slip velocity of the glacier and the shear stress of the bedrock during sliding.

Through this project, I can try to establish the relationship between the factors affecting the friction coefficient and the friction coefficient and understand the importance of different factors to the friction coefficient. At the same time, I can also get the theoretical calculated value of the glacier slip velocity to verify the formula by comparing the result with the actual measurement or experimental data. The **innovation** is to develop a way of thinking about processing collected data with to explore interrelationships and intrinsic connections between these data.

## Resources & Timeline

I have very little code about the project at hand. But I have abundant resources:
  - I’ll be using data of friction coefficients from simulation experiments from publications.
  - I’ll develop further based on existing packages like `nmupy`, `matplotlib`, `mpl_toolkits` and other useful packages.
  - I’ll use docstrings describes source of data and formulas.
  - A lot of work has been done in this field before. There are many code packages on the Internet with powerful data processing functions. These all help me to complete the project in time.
  - I will continue to work on this project in the future to build better model on fitting data.

**Timeline**

I plan to finish this programming project in three weeks. During the first week, I'll focus on finding experimental or field data suitable for the project, and existing code packages that work with those data. In the second week, my goal is to integrate these code packages, redesign the modular structure and build the new functionality needed. In the third week, I will test and validate the designed module and address potential issues to ensure that the module can implement the whole process from data input to data processing.


## Testing, validation, documentation

I will test the accuracy of the fitted relationship with r2 with package `scikit-learn` in linear fit and OLS model. I will fit the data with a neural network to test the relationships fitted by the OLS model. 

In `test_function.py`, I will write some test functions to test the functions I wrote in `my_functions.py` and execute these tests in `Run_test.ipynb`.
