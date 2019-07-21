# -*- coding: utf-8 -*-
"""
Does money make people happier?

Date: 01/06/2019
"""

###############################################################################
## Setup
###############################################################################

# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals

# Common imports
import numpy as np
import os

# to make this notebook's output stable across runs
np.random.seed(42)

# To plot pretty figures
# %matplotlib inline # need for Jupyter notebook
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# Where to save the figures
PROJECT_ROOT_DIR = "C:\\Users\\User\\Documents\\Data Science\\Python\\Hands_on_Machine_Learning\\handson-ml\\Projects"
PROJECT_ID = "lifesat" # name of project

def save_fig(fig_id, tight_layout=True):        # tight_layout automatically adjusts subplot params so that the subplot(s) fits in to the figure area
    path = os.path.join(PROJECT_ROOT_DIR, PROJECT_ID, "images", fig_id + ".png")
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)    # dpi specifies resolution. digits per inch

# Ignore useless warnings (see SciPy issue #5998)
import warnings
warnings.filterwarnings(action="ignore", message="^internal gelsd")

###############################################################################
## Data Prep
###############################################################################

def prepare_country_stats(oecd_bli, gdp_per_capita):
    oecd_bli = oecd_bli[oecd_bli["INEQUALITY"]=="TOT"]
    oecd_bli = oecd_bli.pivot(index="Country", columns="Indicator", values="Value")
    gdp_per_capita.rename(columns={"2015": "GDP per capita"}, inplace=True)
    gdp_per_capita.set_index("Country", inplace=True)
    full_country_stats = pd.merge(left=oecd_bli, right=gdp_per_capita,
                                  left_index=True, right_index=True)
    full_country_stats.sort_values(by="GDP per capita", inplace=True)
    remove_indices = [0, 1, 6, 8, 33, 34, 35] # remove data for now to possibly explore overfitting
    keep_indices = list(set(range(36)) - set(remove_indices))
    return full_country_stats[["GDP per capita", 'Life satisfaction']].iloc[keep_indices]


datapath = os.path.join(PROJECT_ROOT_DIR, PROJECT_ID, "datasets\\")

# Code example
import pandas as pd
import sklearn.linear_model
import sklearn.neighbors # for instance based learning example

# Load the data
oecd_bli = pd.read_csv(datapath + "oecd_bli_2015.csv", thousands=',')
gdp_per_capita = pd.read_csv(datapath + "gdp_per_capita.csv",thousands=',',delimiter='\t',
                             encoding='latin1', na_values="n/a")

# Prepare the data
country_stats = prepare_country_stats(oecd_bli, gdp_per_capita)
X = np.c_[country_stats["GDP per capita"]]
y = np.c_[country_stats["Life satisfaction"]] # Target Variables

# Visualize the data
country_stats.plot(kind='scatter', x="GDP per capita", y='Life satisfaction')
plt.show()

# Select a linear model
model = sklearn.linear_model.LinearRegression()
model2 = sklearn.neighbors.KNeighborsRegressor(n_neighbors=3)

# Train the model
model.fit(X, y)
model2.fit(X, y)

# See parameters of model
t0, t1 = model.intercept_[0], model.coef_[0][0]

country_stats.plot(kind='scatter', x="GDP per capita", y='Life satisfaction')
X=np.linspace(0, 60000, 1000)
plt.plot(X, t0 + t1*X, "b")
save_fig('best_fit_model_plot')
plt.show()

# Make a prediction for Cyprus
X_new = [[22587]]  # Cyprus' GDP per capita
print(model.predict(X_new)) # outputs [[ 5.96242338]]
print(model2.predict(X_new)) # outputs [[5.76666667]]








