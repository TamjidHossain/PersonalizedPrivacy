#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 13:39:03 2021

@author: mdtamjidhossain
"""

import matplotlib.pyplot as plt
import pandas as pd
import glob
import os

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

os.chdir('/Users/mdtamjidhossain/Industry4/Dr Badsha/Arpan/microgrid/')

df = pd.DataFrame()
count = 1
for items in glob.glob("*.csv"):
    print('count: ', count, "Filename: ", items)
    df0 = pd.DataFrame()
    df0 = pd.read_csv(items, names = ['datetime', 'consumption'], delimiter=',')
    df0['HID'] = count
    df = df.append(df0)
    count = count + 1
    
df.datetime = pd.to_datetime(df.datetime, unit= 's')
#%%

# Single House plot - per minute plot

plt.plot(df[df['HID'] == 50]['datetime'], df[df['HID'] == 50]['consumption'], label = 'HID 1 Consumption')
# plt.plot(df['datetime'], df['consumption'], label = 'HID 1 Consumption')
plt.xlabel('Time (minute)', fontweight='bold')
plt.ylabel('Consumption (kW)', fontweight='bold')
plt.xticks(rotation=45)
plt.legend()
plt.grid()
plt.show()

#%%
# grouping by hour and HID

df_grp = pd.DataFrame()
df_grp = df.copy()
df_grp['hour'] = df.datetime.dt.hour
df_grp = df_grp.groupby(['hour', 'HID']).sum()
df_grp = df_grp.reset_index()

#%%
# grouping by HID

df_grp1 = pd.DataFrame()
df_grp1 = df.copy()
df_grp1 = df_grp1.groupby(['HID']).sum()
df_grp1 = df_grp1.reset_index()

#%%
# surface (3D plot) - df_grp

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from sys import argv

x = df_grp.hour
y = df_grp.HID
z = df_grp.consumption

fig = plt.figure()
ax = Axes3D(fig)
surf = ax.plot_trisurf(x, y, z, cmap=cm.jet, linewidth=0.1)
fig.colorbar(surf, shrink=0.5, aspect=5, pad=0.1)
ax.set_xlabel('Hour', fontweight='bold')
ax.set_ylabel('HID', labelpad=7, fontweight='bold')
ax.set_zlabel('Consumption (kW)', labelpad=7, fontweight='bold')
plt.show()

#%%

from scipy.interpolate import make_interp_spline
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as dates
from datetime import datetime


# smooth
date_num_smooth = np.linspace(df_grp1['HID'].min(), df_grp1['HID'].max(), 100) 
spl = make_interp_spline(df_grp1['HID'], df_grp1['consumption'], k=3)
value_np_smooth = spl(date_num_smooth)
# print

plt.plot(date_num_smooth, value_np_smooth)
plt.show()

#%%

# Edge node graphs


df_grp1['DP-consumption0.1'] = df_grp1.apply(lambda row: row['consumption'] + np.random.laplace(loc = 0, scale=1/0.1), axis = 1 )
df_grp1['DP-consumption0.2'] = df_grp1.apply(lambda row: row['consumption'] + np.random.laplace(loc = 0, scale=1/0.2), axis = 1 )
df_grp1['DP-consumption0.3'] = df_grp1.apply(lambda row: row['consumption'] + np.random.laplace(loc = 0, scale=1/0.3), axis = 1 )
df_grp1['DP-consumption0.4'] = df_grp1.apply(lambda row: row['consumption'] + np.random.laplace(loc = 0, scale=1/0.4), axis = 1 )

df_grp1['differences0.1'] = df_grp1['consumption'] - df_grp1['DP-consumption0.1']
df_grp1['differences0.2'] = df_grp1['consumption'] - df_grp1['DP-consumption0.2']
df_grp1['differences0.3'] = df_grp1['consumption'] - df_grp1['DP-consumption0.3']
df_grp1['differences0.4'] = df_grp1['consumption'] - df_grp1['DP-consumption0.4']

plt.plot(df_grp1['HID'][0:100], df_grp1['differences0.1'][0:100], label = 'DP: eps = 0.1')
plt.plot(df_grp1['HID'][0:100], df_grp1['differences0.2'][0:100], label = 'DP: eps = 0.2')
plt.plot(df_grp1['HID'][0:100], df_grp1['differences0.3'][0:100], label = 'DP: eps = 0.3')
plt.plot(df_grp1['HID'][0:100], df_grp1['differences0.4'][0:100], label = 'DP: eps = 0.4')

plt.xlabel('HID', fontweight='bold')
plt.ylabel('Privacy cost', fontweight='bold')
# plt.ylim(0,100000)
# plt.xlim(1,10)
plt.legend(loc='upper right')
plt.xticks(rotation=45)
plt.grid()
plt.show()

df_grp1['differences0.1'].plot.kde(label = 'DP: eps = 0.1')
df_grp1['differences0.2'].plot.kde(label = 'DP: eps = 0.2')
df_grp1['differences0.3'].plot.kde(label = 'DP: eps = 0.3')
df_grp1['differences0.4'].plot.kde(label = 'DP: eps = 0.4')
plt.grid()
plt.xlim(-50,50)
plt.legend()
plt.ylabel('PDF', fontweight='bold')
plt.xlabel('DP Observation', fontweight='bold')
plt.show()
#%%

# Fog Nodes - 1

df_grp2 = df_grp1[0:100].copy()

df_grp2['normalize'] = df_grp2['consumption']/df_grp2['consumption'].max()

def DP(normalize, eps):
    noise = np.random.laplace(loc = 0, scale=1/eps)
    normalize1 = normalize + noise
    while normalize1 < 0:
        noise = np.random.laplace(loc = 0, scale=1/eps)
        normalize1 = normalize + noise
    return normalize1
    


df_grp2['normalize0.6'] = df_grp2.apply(lambda row: DP(row['normalize'], 0.6), axis = 1 )
df_grp2['normalize0.7'] = df_grp2.apply(lambda row: DP(row['normalize'], 0.7), axis = 1 )


plt.plot(df_grp2['HID'], df_grp2['normalize'], label = 'no DP', color = 'red')
plt.plot(df_grp2['HID'], df_grp2['normalize0.6'], label = 'DP: eps = 0.6')
plt.plot(df_grp2['HID'], df_grp2['normalize0.7'], label = 'DP: eps = 0.7')


plt.xlabel('HID', fontweight='bold')
plt.ylabel('Normalized consumption (kW)', fontweight='bold')
# plt.ylim(0,100000)
# plt.xlim(1,10)
plt.legend(loc = 'upper right')
plt.xticks(rotation=45)
plt.grid()
plt.show()






# df_grp2['DP-consumption0.6'] = df_grp2['normalize0.6'] * df_grp2['consumption'].max()
# df_grp2['DP-consumption0.7'] = df_grp2['normalize0.7'] * df_grp2['consumption'].max()


# plt.plot(df_grp2['HID'], df_grp2['normalize'], label = 'no DP', color = 'red')
# plt.plot(df_grp2['HID'], df_grp2['DP-consumption0.6'], label = 'DP: eps = 0.6')
# plt.plot(df_grp2['HID'], df_grp2['DP-consumption0.7'], label = 'DP: eps = 0.7')


# plt.xlabel('HID', fontweight='bold')
# plt.ylabel('Consumption (kW)', fontweight='bold')
# # plt.ylim(0,100000)
# # plt.xlim(1,10)
# plt.legend()
# plt.xticks(rotation=45)
# plt.grid()
# plt.show()

