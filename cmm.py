# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 21:13:09 2018

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

x_ar = x_df.as_matrix(columns=None)
y_ar = y_df.as_matrix(columns=None)

#Now fit Y Slice
popt, pcov = curve_fit(func1, y_ar, x_ar, bounds=(0, [200., 2000., 500.]))

#Plot both raw data and the hyperbolic fit to the y-slice
plot1 = plt.figure(1)
plt.xlabel('Y(mm)')
plt.ylabel('X(mm)')
plt.title('Y slice Confocal Probe & CMM Primary Mirror')
plt.plot(y_ar, x_ar, 'bo')
y_fit = np.arange(250,600,0.1)
plt.plot(y_fit, np.sqrt(popt[0]**2*(1+((y_fit-popt[2])**2/popt[1]**2))),'r-')
plt.ylim([87,95])
print('Y-slice: a=', popt[0],'b=', popt[1],'c=', popt[2])

#Now lets find the minimum X & Y position of the fit
func1_d =np.sqrt(popt[0]**2*(1+((y_fit-popt[2])**2/popt[1]**2)))
func1_d_matr = np.dstack((y_fit,func1_d))
minima1 = np.argmin(func1_d_matr, axis=1)
minima1y = y_fit[minima1[0,1]]
minima1x = func1_d[minima1[0,1]]
print('Y minimum is at Y=', minima1y, 'and X=', minima1x )
plt.plot([minima1y], [minima1x], 'ro')
plt.legend(('raw', 'fit', 'minimum'), loc='upper center', shadow=False)


#Now lets look at the X slice raw data and convert it to array format
y2_df = z['CMM X.1']
x2_df = z['Conf total z.1']
x2_df = x2_df.iloc[0:35] #locate  the end as NaN beyond this in the column of data
y2_df = y2_df.iloc[0:35]

x2_ar = x2_df.as_matrix(columns=None)
y2_ar = y2_df.as_matrix(columns=None)

#Now fit X Slice
popt2, pcov2 = curve_fit(func2, y2_ar, x2_ar, bounds=(0, [200., 2000., 1000.]))

#Plot both raw data and the hyperbolic fit to the x-slice
plot2 = plt.figure(2)
plt.xlabel('Y(mm)')
plt.ylabel('X(mm)')
plt.title('X slice Confocal Probe & CMM Primary Mirror')
y_fit2 = np.arange(75,400,0.1)
plt.plot(y2_ar, x2_ar, 'bo')
plt.plot(y_fit2, np.sqrt(popt2[0]**2*(1+((y_fit2-popt2[2])**2/popt2[1]**2))),'r-')
print('X-slice: a=', popt2[0],'b=', popt2[1],'c=', popt2[2])

#Now lets find the minimum X & Y position of the fit
func2_d =np.sqrt(popt2[0]**2*(1+((y_fit2-popt2[2])**2/popt2[1]**2)))
func2_d_matr = np.dstack((y_fit2,func2_d))
minima2 = np.argmin(func2_d_matr, axis=1)
minima2y = y_fit2[minima2[0,1]]
minima2x = func1_d[minima2[0,1]]
print('Y minimum is at Y=', minima2y, 'and X=', minima2x )
plt.plot([minima2y], [minima2x], 'ro')
plt.legend(('raw', 'fit', 'minimum'), loc='upper center', shadow=False)

'''
To test plot hyperbola with defined coefficients
#Hyperbola
a=88
b=1000
c=250
x = np.linspace(100, 400, 1000)
y = np.sqrt(a**2*(1+((x-c)**2/b**2)))

#Plot general hyperbola
plot3 = plt.figure(3)
plt.xlabel('Y(mm)')
plt.ylabel('X(mm)')
plt.title('Hyperbola')
plt.plot(x, y, 'bo')
'''
#Now table CMM confocal probe measurements
cmm_table_x_df = z['Table CMM X']
cmm_table_y_df = z['Table CMM Y']
cmm_table_z_df = z['Table Confocal Probe Total Z']
cmm_table_x_df = cmm_table_x_df.iloc[0:6]
cmm_table_y_df = cmm_table_y_df.iloc[0:6]
cmm_table_z_df = cmm_table_z_df.iloc[0:6]
cmm_table_x = cmm_table_x_df.as_matrix(columns=None)
cmm_table_y = cmm_table_y_df.as_matrix(columns=None)
cmm_table_z = cmm_table_z_df.as_matrix(columns=None)

#Now fit plane
Xcmm_table,Ycmm_table = np.meshgrid(np.arange(100, 600, 50), np.arange(100, 600, 50))
XX = Xcmm_table.flatten()
YY = Ycmm_table.flatten()

Acmm_table = np.c_[cmm_table_x, cmm_table_y, np.ones(cmm_table_x.shape[0])]
Ccmm_table,_,_,_ = scipy.linalg.lstsq(Acmm_table, cmm_table_z)    # coefficients
# evaluate it on grid
Zcmm_table = Ccmm_table[0]*Xcmm_table + Ccmm_table[1]*Ycmm_table + Ccmm_table[2]

plot3 = plt.figure(3)
ax=plot3.gca(projection='3d')
plt.title('CMM Granite Table')
ax.plot_wireframe(Xcmm_table, Ycmm_table, Zcmm_table)
ax.scatter(cmm_table_x, cmm_table_y, cmm_table_z, c='r', marker='o')
ax.set_xlabel('CMM X')
ax.set_ylabel('CMM Y')
ax.set_zlabel('CMM Confocal Totoal Z')

#Error between raw data and fit
Zcmm_table_fit = Ccmm_table[0]*cmm_table_x + Ccmm_table[1]*cmm_table_y + Ccmm_table[2] #values of the fit at the measured X & Y locations
Zcmm_table_diff = Zcmm_table_fit - cmm_table_z #difference
Zcmm_table_diff_mean = np.mean(Zcmm_table_diff)
print('Mean error of the fit=', Zcmm_table_diff_mean)
Zcmm_table_diff_std = np.std(Zcmm_table_diff)
print('Standard deviation of the fit=', Zcmm_table_diff_std)

'''
Now look at other CMM measurements
'''
#input csv file
cmm_df = pd.read_csv("fc2.csv", encoding = "ISO-8859-1")

#Outer diameter of the primary mirror
outer_d_primary_x_df = cmm_df['CIRCLE'].iloc[9:17]
outer_d_primary_y_df = cmm_df['CR4'].iloc[9:17]
outer_d_primary_z_df = cmm_df['CART'].iloc[9:17]
outer_d_primary_x = np.array(outer_d_primary_x_df, dtype=np.float64)
outer_d_primary_y = np.array(outer_d_primary_y_df, dtype=np.float64)
outer_d_primary_z = np.array(outer_d_primary_z_df, dtype=np.float64)

#Inner cyclinder of the primary mirror
inner_d_primary_x_df = cmm_df['CIRCLE'].iloc[29:37]
inner_d_primary_y_df = cmm_df['CR4'].iloc[29:37]
inner_d_primary_z_df = cmm_df['CART'].iloc[29:37]
inner_d_primary_x = np.array(inner_d_primary_x_df, dtype=np.float64)
inner_d_primary_y = np.array(inner_d_primary_y_df, dtype=np.float64)
inner_d_primary_z = np.array(inner_d_primary_z_df, dtype=np.float64)

#Table position with the CMM
table_cmm_x_df = cmm_df['CIRCLE'].iloc[48:57]
table_cmm_y_df = cmm_df['CR4'].iloc[48:57]
table_cmm_z_df = cmm_df['CART'].iloc[48:57]
table_cmm_x = np.array(table_cmm_x_df, dtype=np.float64)
table_cmm_y = np.array(table_cmm_y_df, dtype=np.float64)
table_cmm_z = np.array(table_cmm_z_df, dtype=np.float64)

#Primary mirror surface on the edges
primary_surface_x_df = cmm_df['CIRCLE'].iloc[68:77]
primary_surface_y_df = cmm_df['CR4'].iloc[68:77]
primary_surface_z_df = cmm_df['CART'].iloc[68:77]
primary_surface_x = np.array(primary_surface_x_df, dtype=np.float64)
primary_surface_y = np.array(primary_surface_y_df, dtype=np.float64)
primary_surface_z = np.array(primary_surface_z_df, dtype=np.float64)

#Plot these
plot4 = plt.figure(4)
ax=plot4.gca(projection='3d')
plt.title('CMM Measurements')
#ax.plot_wireframe(X, Y, Z)
ax.scatter(outer_d_primary_x, outer_d_primary_y, outer_d_primary_z, c='r', marker='o')
ax.scatter(inner_d_primary_x, inner_d_primary_y, inner_d_primary_z, c='b', marker='o')
ax.scatter(table_cmm_x, table_cmm_y, table_cmm_z, c='g', marker='o')
ax.scatter(primary_surface_x, primary_surface_y, primary_surface_z, c='y', marker='o')
plt.legend(('Primary Mirror OD', 'Primary Mirror Inner Cylinder', 'Granite Table', 'Primary Mirror Optical Surface'), loc='upper right', shadow=False)
ax.set_xlabel('CMM X')
ax.set_ylabel('CMM Y')
ax.set_zlabel('CMM Z')

#Now fit planes
Xtable_cmm,Ytable_cmm = np.meshgrid(np.arange(100, 600, 50), np.arange(100, 600, 50))
XX = Xtable_cmm.flatten()
YY = Ytable_cmm.flatten()
Atable_cmm = np.c_[table_cmm_x, table_cmm_y, np.ones(table_cmm_x.shape[0])]
Ctable_cmm,_,_,_ = scipy.linalg.lstsq(Atable_cmm, table_cmm_z)    # coefficients

Xodp,Yodp = np.meshgrid(np.arange(100, 600, 50), np.arange(100, 600, 50))
XX = Xodp.flatten()
YY = Yodp.flatten()
Aodp = np.c_[outer_d_primary_x, outer_d_primary_y, np.ones(outer_d_primary_x.shape[0])]
Codp,_,_,_ = scipy.linalg.lstsq(Aodp, outer_d_primary_z)    # coefficients

Xidp,Yidp = np.meshgrid(np.arange(100, 600, 50), np.arange(100, 600, 50))
XX = Xidp.flatten()
YY = Yidp.flatten()
Aidp = np.c_[inner_d_primary_x, inner_d_primary_y, np.ones(inner_d_primary_x.shape[0])]
Cidp,_,_,_ = scipy.linalg.lstsq(Aidp, inner_d_primary_z)    # coefficients  

Xps,Yps = np.meshgrid(np.arange(100, 600, 50), np.arange(100, 600, 50))
XX = Xps.flatten()
YY = Yps.flatten()
Aps = np.c_[primary_surface_x, primary_surface_y, np.ones(primary_surface_x.shape[0])]
Cps,_,_,_ = scipy.linalg.lstsq(Aps, primary_surface_z)    # coefficients                               

                                     
# evaluate it on grid
Ztable_cmm = Ctable_cmm[0]*Xtable_cmm + Ctable_cmm[1]*Ytable_cmm + Ctable_cmm[2]
Zodp = Codp[0]*Xodp + Codp[1]*Yodp + Codp[2]
Zidp = Cidp[0]*Xidp + Cidp[1]*Yidp + Cidp[2]
Zps = Cps[0]*Xps + Cps[1]*Yps + Cps[2]

#plots
ax.plot_wireframe(Xtable_cmm, Ytable_cmm, Ztable_cmm)
ax.plot_wireframe(Xodp, Yodp, Zodp)
ax.plot_wireframe(Xidp, Yidp, Zidp)
ax.plot_wireframe(Xps, Yps, Zps)

