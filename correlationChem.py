import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model
from misc_utility.plot_utility import *
from misc_utility.solvers import *


INPUT_DIR = "/home/samuel/Documents/IGE/BdD/BdD_PO/"
name_File="CHEM_conc.csv"
colOK   = ("OC","EC",\
           "Na+","NH4+","K+","Mg2+","Ca2+","MSA","Cl-","NO3-","SO42-","Ox",\
           "Levoglucosan","ΣPolyols",\
           "As","Ba","Cd","Cu","Mn","Mo","Ni","Pb","Sb","Sn","Ti","Zn","Al","Fe",\
           "ΣHAP_part","ΣHOP","PM10")
colOK = ("Levoglucosan","ΣPolyols","MSA","Cu","Fe","Ox","NO3-","SO42-","ΣHOP","EC")
keep = ["DTTv"]

name = "Chamonix"
ChamonixCHEM    = pd.read_csv(INPUT_DIR+name+"/"+name+name_File,
                          index_col="date", parse_dates=["date"])
ChamonixPO      = pd.read_csv(INPUT_DIR+name+"/"+name+"PO.csv",
                              index_col="date", parse_dates=["date"])
# ChamonixPO = ChamonixPO[keep]
name = "Marnaz"
MarnazCHEM    = pd.read_csv(INPUT_DIR+name+"/"+name+name_File,
                          index_col="date", parse_dates=["date"])
MarnazPO      = pd.read_csv(INPUT_DIR+name+"/"+name+"PO.csv",
                              index_col="date", parse_dates=["date"])
# MarnazPO = MarnazPO[keep]
name = "Passy"
PassyCHEM    = pd.read_csv(INPUT_DIR+name+"/"+name+name_File,
                          index_col="date", parse_dates=["date"])
PassyPO      = pd.read_csv(INPUT_DIR+name+"/"+name+"PO.csv",
                              index_col="date", parse_dates=["date"])
# PassyPO = PassyPO[keep]


TMPMarnaz = pd.concat([MarnazCHEM.ix[:,colOK],MarnazPO.ix[:,keep]],axis=1,join="inner")
TMPChamonix = pd.concat([ChamonixCHEM.ix[:,colOK],ChamonixPO.ix[:,keep]],axis=1,join="inner")
TMPPassy = pd.concat([PassyCHEM.ix[:,colOK],PassyPO.ix[:,keep]],axis=1,join="inner")

corrMarnaz = TMPMarnaz.corr().ix[:,keep]
corrPassy = TMPPassy.corr().ix[:,keep]
corrChamonix = TMPChamonix.corr().ix[:,keep]

# f, axarr = plt.subplots(1,3, sharey=True)
f = plt.figure()
plt.subplot(131)
plt.imshow(corrChamonix,cmap='RdBu_r',vmin=-1,vmax=1)
ax = plt.gca()
ax.set_yticks(range(len(corrChamonix.index)))
ax.set_yticklabels(corrChamonix.index)
plt.subplot(132)
plt.imshow(corrMarnaz,cmap='RdBu_r',vmin=-1,vmax=1)
plt.subplot(133)
plt.imshow(corrPassy,cmap='RdBu_r',vmin=-1,vmax=1)

f.subplots_adjust(wspace=0)
[a.set_yticks([]) for a in f.axes[1:]];
[a.set_xticks(range(len(corrChamonix.columns))) for a in f.axes]
[a.set_xticklabels(corrChamonix.columns, rotation=90) for a in f.axes]
plt.colorbar()

TMPChamonix.dropna(inplace=True)
TMPMarnaz.dropna(inplace=True)
TMPPassy.dropna(inplace=True)
Chamonix = solve_scikit_linear_regression(TMPChamonix.ix[:,colOK].as_matrix(),
                                          TMPChamonix.ix[:,keep].as_matrix(),
                                          index=colOK)
Marnaz = solve_scikit_linear_regression(TMPMarnaz.ix[:,colOK].as_matrix(),
                                        TMPMarnaz.ix[:,keep].as_matrix(),
                                        index=colOK)
Passy = solve_scikit_linear_regression(TMPPassy.ix[:,colOK].as_matrix(),
                                       TMPPassy.ix[:,keep].as_matrix(),
                                       index=colOK)

Chamonix.name = "Chamonix"
Marnaz.name = "Marnaz"
Passy.name = "Passy"

dfall = pd.concat([Chamonix,Marnaz,Passy],axis=1)
print(dfall)
dfall.plot.bar()
ax = plt.gca()
ax.set_yscale('log')
