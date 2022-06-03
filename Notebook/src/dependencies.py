"""Import dependencies for this notebook"""

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
