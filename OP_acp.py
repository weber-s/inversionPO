import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from customClass import *
from misc_utility.plot_utility import *
from misc_utility.load_data import *
from sklearn.decomposition import PCA

INPUT_DIR = "/home/samuel/Documents/IGE/BdD/BdD_OP/"

list_station= ["ANDRA","Nice","Frenes","Passy","Chamonix",
               "Marnaz","Marseille","PdB"]
# list_station= ["Frenes",]

list_OPtype = ["AAv","DTTv"]

# format to save plot
fmt_save    =["png","pdf","svg"]


plt.interactive(False)

# Choose the inversion method (could be OLS, WLS, GLS or ML)
inversion_method = "WLS"
OUTPUT_DIR="/home/samuel/Documents/IGE/inversionOP/figures/inversion"+inversion_method+"_wo_outliers/"
SAVE_DIR="/home/samuel/Documents/IGE/inversionOP/results/inversion"+inversion_method+"_wo_outliers/"

fromSource  = True

# sort list in order to always have the same order
list_station.sort()
list_OPtype.sort()
# initialize stuff
sto = dict()
for OPtype in list_OPtype:
    sto[OPtype]=dict()
    print("=============="+OPtype+"====================")
    pie = pd.Series()
    for name in list_station:
        print("=============="+name+"====================")
        CHEM = load_CHEMorOP(name, INPUT_DIR, CHEMorOP="CHEM", fromSource=fromSource)
        OP   = load_CHEMorOP(name, INPUT_DIR, CHEMorOP="OP")

        # rename columns
        CHEM = setSourcesCategories(CHEM)
        CHEM.sort_index(axis=1, inplace=True)


        if not(OPtype in OP.columns) or OP[OPtype].isnull().all():
            sto[OPtype][name] = Station(name=name, CHEM=CHEM, hasOP=False)
            pie = pd.concat([pie, sto[OPtype][name].m],axis=1)
            continue

        # ==== Drop days with missing values
        TMP = pd.DataFrame.join(CHEM,OP[[OPtype,"SD_"+OPtype]],how="inner")
        TMP.dropna(inplace=True)
        OP   = TMP[OPtype]
        OPunc= TMP["SD_"+OPtype]
        CHEM = TMP[CHEM.columns]
        OP.name = OPunc.name = CHEM.name = name
        
        TMP.drop(["date","SD_"+OPtype], axis=1, inplace=True)

        # ==== 
        pca = PCA(n_components=2)
        X_r = pca.fit(TMP).transform(TMP)
        print(X_r)
