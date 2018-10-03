import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import sklearn
from scipy import linalg, polyfit
#import pulp
import pickle
from sklearn import linear_model
from customClass import *
from misc_utility.plot_utility import *
from misc_utility.save_utility import result2csv
from misc_utility.solvers import *
from misc_utility.load_data import *
from statsmodels.sandbox.regression.predstd import wls_prediction_std

sns.reset_defaults()

plt.interactive(True)

INPUT_DIR = "/home/samuel/Documents/IGE/BdD/BdD_OP/"
OUTPUT_DIR= "/home/samuel/Documents/IGE/inversionOP/figures/inversionWLS/"
SAVE_DIR  = "/home/samuel/Documents/IGE/inversionOP/results/inversionWLS/"
# list_station= ["Nice","Frenes","Passy","Chamonix", "Marnaz"]
list_station= ["ANDRA","PdB","Marseille","Nice","Frenes","Passy","Chamonix", "Marnaz"]

list_OPtype = ["AAv","DTTv"]

list_station.sort()
list_OPtype.sort()

OP = dict()
for name in list_station:
    OP[name] = load_CHEMorOP(name, INPUT_DIR, CHEMorOP="OP")

colors = sitesColor()
colors = colors.ix["color", list_station].values
f, axes = plt.subplots(nrows=1, ncols=2,sharey=True,figsize=(8.74,3.1))
for i, OPtype in enumerate(list_OPtype):
    for j,name in enumerate(list_station):
        axes[i].scatter(OP[name][OPtype], OP[name]["SD_"+OPtype], label=name,
                        s=5,
                        color=colors[j])
        axes[i].set_xlabel(OPtype)
    
    if i ==0:
        axes[i].set_ylabel('Standard error')

plt.subplots_adjust(top=0.93, bottom=0.16, left=0.12, right=0.81)
l = plt.legend(loc='center', bbox_to_anchor=(1.3, 0.5),
               ncol=1, fancybox=True)

# f, axes = plt.subplots(nrows=1, ncols=2,sharey=True,figsize=(8.74,3.1))
# for i, OPtype in enumerate(list_OPtype):
#     for name in list_station:
#         OP[name]["SD_"+OPtype].plot.density(ax=axes[i], label=name)
#         axes[i].set_xlabel(OPtype)
#     
#     if i ==0:
#         axes[i].set_ylabel('Standard error')
#
# plt.subplots_adjust(top=0.93, bottom=0.16, left=0.12, right=0.81)
# l = plt.legend(loc='center', bbox_to_anchor=(1.3, 0.5),
#                ncol=1, fancybox=True)
