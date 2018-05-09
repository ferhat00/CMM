# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 16:28:51 2018

@author: fculfaz
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sympy import *
import pandas as pd
from scipy.optimize import curve_fit
import scipy.linalg
from matplotlib import cm
from scipy.misc import derivative

def x(xp,yp,theta):
        return xp*np.cos(theta) + yp*np.sin(theta)
    
def y(xp,yp,theta):
        return -xp*np.sin(theta) + yp*np.cos(theta)    

def hyp(x,a,b,h,k):
        return k + np.sqrt((a**2)*(1+(((x-h)**2)/b**2)))

fname = 'FC_confocal_gt_cmm.csv'
z = pd.read_csv("FC_confocal_gt_cmm.csv", encoding = "ISO-8859-1")

#Now lets look at the Y slice raw data and convert it to array format

cmm_y_df = z['CMM Y']
cmm_z_y_df = z['Conf total z']

cmm_z_y = cmm_z_y_df.as_matrix(columns=None)
cmm_y = cmm_y_df.as_matrix(columns=None)

#Now fit Y Slice
popt_y, pcov_y = curve_fit(hyp, cmm_y, cmm_z_y, bounds=(0,[200.,2000.,500.,1000.]))

#Plot Y slice
plot1 = plt.figure(1)
plt.xlabel('Y(mm)')
plt.ylabel('X(mm)')
plt.title('Y slice Confocal Probe & CMM Primary Mirror')
plt.plot(cmm_y, cmm_z_y, 'bo')
y_fit = np.arange(200,600,0.1)
x_fit = np.arange(0,350,0.1)
fit = (popt_y[3] + np.sqrt((popt_y[0]**2)*(1+(((y_fit-popt_y[2])**2)/popt_y[1]**2))))
plt.plot(y_fit, fit,'r-')
print('Y-slice: a=', popt_y[0],'b=', popt_y[1],'h=', popt_y[2], 'k=',popt_y[3])

#Find and plot the minimum
fit_stack = np.dstack((y_fit,fit))
fit_mat = np.dstack((y_fit,fit_stack))
minima1 = np.argmin(fit_stack, axis=1)
minima1y = y_fit[minima1[0,1]]
minima1x = fit[minima1[0,1]]
print('Y minimum is at Y=', minima1y, 'and X=', minima1x )
plt.plot([minima1y], [minima1x], 'go')
plt.legend(('raw', 'fit', 'minimum'), loc='upper center', shadow=False)

#--------------------------------------------------------------------------------

#Now lets look at the X slice raw data and convert it to array format
cmm_x_df = z['CMM X.1']
cmm_z_x_df = z['Conf total z.1']
cmm_z_x = cmm_z_x_df.iloc[0:35].as_matrix(columns=None) #locate  the end as NaN beyond this in the column of data
cmm_x = cmm_x_df.iloc[0:35].as_matrix(columns=None)

#Now fit X Slice
popt_x, pcov_x = curve_fit(hyp, cmm_x, cmm_z_x, bounds=(0,[200.,2000.,500.,1000.]))

#Plot X slice
plot2 = plt.figure(2)
plt.xlabel('Y(mm)')
plt.ylabel('X(mm)')
plt.title('X slice Confocal Probe & CMM Primary Mirror')
plt.plot(cmm_x, cmm_z_x, 'bo')
y_fit2 = np.arange(50,450,0.1)
#x_fit = np.arange(0,350,0.1)
fit2 = (popt_x[3] + np.sqrt((popt_x[0]**2)*(1+(((y_fit2-popt_x[2])**2)/popt_x[1]**2))))
plt.plot(y_fit2, fit2,'r-')
print('X-slice: a=', popt_x[0],'b=', popt_x[1],'h=', popt_x[2], 'k=',popt_x[3])

#Find and plot the minimum
fit_stack2 = np.dstack((y_fit2,fit2))
fit_mat2 = np.dstack((y_fit2,fit_stack2))
minima2 = np.argmin(fit_stack2, axis=1)
minima2y = y_fit2[minima2[0,1]]
minima2x = fit2[minima2[0,1]]
print('X minimum is at Y=', minima2y, 'and X=', minima2x )
plt.plot([minima2y], [minima2x], 'go')
plt.legend(('raw', 'fit', 'minimum'), loc='upper center', shadow=False)

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

#Now lets fit circles to cloud of points
def circ(x_circ,a_circ,b_circ,r):
        return np.sqrt((r**2)-((x_circ-a_circ)**2)) + b_circ

popt_circ, pcov_circ = curve_fit(circ, outer_d_primary_x, outer_d_primary_y, bounds=(100,[400.,400.,500.]))

fit_circ = np.sqrt((popt_circ[2]**2)-((x_fit-popt_circ[0])**2)) + popt_circ[1]

plot5 = plt.figure(5)
plt.plot(outer_d_primary_x, outer_d_primary_y, 'bo')
plt.plot(x_fit, fit_circ, 'r-')


centroid_table_cmm = [np.mean(table_cmm_x),np.mean(table_cmm_y),np.mean(table_cmm_z)]
centroid_outer_d_primary = [np.mean(outer_d_primary_x),np.mean(outer_d_primary_y),np.mean(outer_d_primary_z)]
centroid_inner_d_primary = [np.mean(inner_d_primary_x),np.mean(inner_d_primary_y),np.mean(inner_d_primary_z)]
centroid_primary_surface = [np.mean(primary_surface_x),np.mean(primary_surface_y),np.mean(primary_surface_z)]
print('Centroid outer diameter of primary: X=', centroid_outer_d_primary[0], 'Y=',centroid_outer_d_primary[1], 'Z=',centroid_outer_d_primary[2])
print('Centroid inner diameter of primary: X=', centroid_inner_d_primary[0], 'Y=',centroid_inner_d_primary[1], 'Z=',centroid_inner_d_primary[2])
print('Centroid primary surface: X=', centroid_primary_surface[0], 'Y=',centroid_primary_surface[1], 'Z=',centroid_primary_surface[2])