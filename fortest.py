# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 23:19:19 2018

@author: Ferhat
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from scipy import optimize
from mpl_toolkits.mplot3d import Axes3D
import scipy.linalg
from matplotlib import cm
from sympy import symbols, diff


#define a generic hyperbola function 
def func1(x, a, b, c):
    return np.sqrt(a**2*(1+((x-c)**2/b**2)))

def func2(x2, a2, b2, c2):
    return np.sqrt(a2**2*(1+((x2-c2)**2/b2**2)))

def func3(xp, yp, ap, bp, cp):
    return ap*xp + bp*yp + cp

#Input the raw data file from csv format, read it into a Pandas Dataframe
fname = 'FC_confocal_gt_cmm.csv'
z = pd.read_csv("FC_confocal_gt_cmm.csv", encoding = "ISO-8859-1")

#Now lets look at the Y slice raw data and convert it to array format

y_df = z['CMM Y']
x_df = z['Conf total z']

i=1
for i in range (x,y): 

    (i)_ar = (i)_df.as_matrix(columns=None)

