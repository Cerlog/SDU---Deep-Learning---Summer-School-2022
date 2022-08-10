# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 10:25:31 2022

@author: baeks
"""

import pandas as pd
import numpy as np
import matplotlib as mpl
import statsmodels.api as sm
import matplotlib.pyplot as plt

df = pd.read_csv('auto.csv')

#%%
df_clean = df[df.horsepower != '?']

#%%
df_int = df_clean[['mpg','horsepower','weight','model_year','displacement','origin','acceleration','cylinders']].apply(pd.to_numeric)

df_int[['weight','displacement','horsepower','acceleration']] = np.log(df_int[['weight','displacement','horsepower','acceleration']])

#%%
names = list(df_clean.columns)
for i in range(1,8):
    fig, ax = plt.subplots()  # Create a figure containing a single axes.
    ax.set_ylabel(names[i])
    ax.scatter(df_int[names[0]],df_int[names[i]]);  # Plot some data on the axes.

#%%
Y = df_int['mpg']
X = np.asarray(df_int[['weight','model_year','origin','horsepower','acceleration']])
X = sm.add_constant(X)

mod = sm.OLS(Y,X)

results = mod.fit()

print(results.params)
print(results.summary())
print(results.cov_params())
fig = sm.graphics.plot_partregress_grid(results)
fig.tight_layout(pad=1.0)
