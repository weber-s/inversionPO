import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from sklearn.model_selection import train_test_split
from customClass import *
from itertools import product
from misc_utility.plot_utility import *
from misc_utility.save_utility import *
from misc_utility.solvers import *
from misc_utility.load_data import *



INPUT_DIR = "/home/webersa/Documents/BdD/BdD_OP/"

# list_station=["Nice","Aix","GRE-fr", "Chamonix","Roubaix","STG-cle",
#               "MRS-5av","PdB","Nogent","Talence"]
list_station= ["Chamonix"]

# list_OPtype = ["AAv","DTTv","DCFHv"]
list_OPtype = ["AAv","DTTv"]

# format to save plot
fmt_save    =["png","pdf","svg"]


plt.interactive(True)

# Choose the inversion method (could be OLS, WLS, GLS or ML)
inversion_method = "WLS"
OUTPUT_DIR="/home/webersa/Documents/inversionOP/figures/inversion"+inversion_method+"_wo_outliers/"
SAVE_DIR="/home/webersa/Documents/inversionOP/results/inversion"+inversion_method+"_wo_outliers/"

# OUTPUT_DIR="/home/webersa/Documents/IGE/inversionOP/figures/inversion"+inversion_method+"_DECOMBIO/"
# SAVE_DIR="/home/webersa/Documents/IGE/inversionOP/results/inversion"+inversion_method+"_DECOMBIO/"

fromSource  = True
saveFig     = False
plotTS      = False
plotBar     = False
plotRegplot = False
saveResult  = False
sum_sources = True
plotAll     = False

# Number of bootstrap run
NBOOT = 50
BDIROP = "~/Documents/BdD/BdD_OP/"
BDIRSRC= "~/Documents/BdD/BdD_OP/"

# sort list in order to always have the same order
list_station.sort()
list_OPtype.sort()

# initialize stuff
sto = dict()
sources = list() # list of all the sources

OPandStation = product(list_OPtype, list_station)
# for OPtype, name in OPandStation:
for name in list_station:
    print(name)
    station = Station(name=name, inputDir=INPUT_DIR,
                      SRCfile="{BDIRSRC}/{site}/{site}_SRC_Florie_BCwb.csv".format(BDIRSRC=BDIRSRC,site=name),
                      OPfile="{BDIROP}/{site}/{site}_OP.csv".format(BDIROP=BDIROP,
                                                                    site=name)
                      ,list_OPtype=list_OPtype)
    station.load_SRC()
    station.load_OP()
    station.setSourcesCategories()
    station.SRC.sort_index(axis=1, inplace=True)
    station.OPi = pd.DataFrame(index=station.SRC.columns, columns=list_OPtype)
    if sum_sources:
        station.mergeSources(inplace=True)
    for OPtype in list_OPtype:
        print(name,OPtype)
        if not(OPtype in station.OP.columns) or station.OP[OPtype].isnull().all():
            # we didn't measure this OP
            # so save it and continue
            print(name)
            print(OPtype)            
            continue

        # ==== Drop days with missing values
        TMP     = station.SRC.merge(station.OP[[OPtype,"SD_"+OPtype]], left_index=True,
                                    right_index=True, how="inner")
        if len(TMP)==0:
            print("WARNING: No commun index")
            continue
        TMP.dropna(inplace=True)
        OP      = TMP[OPtype]
        OPunc   = TMP["SD_"+OPtype]
        CHEM    = TMP[station.SRC.columns]
        OP.name = OPunc.name = CHEM.name = name

        # ==== Different inversion method
        # Here we choose the WLS
        regr    = solve_WLS(X=CHEM, y=OP, sigma=1/OPunc**2)
        station.reg[OPtype] = regr
        # print(regr.summary())
        station.OPi.loc[:,OPtype] = regr.params[1:]
        station.OPi.loc[:,"SD_"+OPtype] = regr.bse[1:]

        # Bootstrap the solution in order to estimate the model uncertainties
        pred = pd.DataFrame(index=CHEM.index)
        for i in list(range(0,NBOOT)):
            params = regr.bse * np.random.randn(len(regr.params))+regr.params
            pred[i] = (params*CHEM).sum(axis=1) + params["const"]
        station.OPmodel_unc[OPtype] = pred.std(axis=1)
        station.OPmodel[OPtype] = regr.params["const"]+(station.SRC * station.OPi[OPtype]).sum(axis=1)
        # station.get_ODR_result(OPtype=OPtype)
        # station.get_pearson_r(OPtype)
        # ==== Store the result
    sto[name] = station

    if saveResult:
        with open(SAVE_DIR+"/"+name+OPtype+".pickle", "wb") as handle:
            pickle.dump(sto[OPtype][name], handle, protocol=pickle.HIGHEST_PROTOCOL)


# sources = setSourcesCategories(sources).sort()
# multiIndex = pd.MultiIndex.from_product([list_OPtype,sources],
#                                         names=["OPtype","Sources"])
# saveCoeff = pd.DataFrame(data=0.0, columns=list_station, index=multiIndex)
# saveCovm  = pd.DataFrame(data=0.0, columns=list_station, index=multiIndex)
# for OPtype in list_OPtype:
#     df =  pd.concat([sto[OPtype][name].OPi for name in list_station], axis=1)
#     df["Sources"] = df.index
#     saveCoeff.loc[OPtype][:]= df.set_index("Sources",drop=True)
#     df  = pd.concat([sto[OPtype][name].covm for name in list_station], axis=1)
#     df["Sources"] = df.index
#     saveCovm.loc[OPtype][:]= df.set_index("Sources",drop=True)
#
if plotTS or plotBar:
    for OPtype, name in product(list_OPtype, list_station):
        if saveResult:
            with open(SAVE_DIR+"/"+name+OPtype+".pickle", "rb") as file:
                station = pickle.load(file)
        else:
            station = sto[OPtype][name]
        if plotTS:
            plot_station(station,OPtype)
            if saveFig:
                plot_save("inversion"+name+OPtype, OUTPUT_DIR, fmt=fmt_save)
        if plotBar:
            plot_ts_reconstruction_OP(station,OPtype)
            if saveFig:
                plot_save("reconstructionPerSource_"+name+OPtype, OUTPUT_DIR,
                          fmt=fmt_save)
if plotRegplot:
    for name in list_station:
        X = sto[list_OPtype[0]][name].CHEM
        Y = pd.DataFrame(columns=list_OPtype)
        for OPtype in list_OPtype:
            Y[OPtype] = sto[OPtype][name].OP
        plot_regplot(X, Y, title=name)
        if saveFig:
            plot_save("regplot_"+name, OUTPUT_DIR, fmt=fmt_save)

if plotAll:
    # ========== CONTRIBUTION PIE CHART ===========================================
    f,axes = plt.subplots(nrows=len(list_OPtype),ncols=len(list_station),figsize=(17,8))
    #ax.shape = (np.sum(ax.shape),) 
    for j, row in enumerate(axes):
        for i, ax in enumerate(row):
            station=sto[list_OPtype[j]][list_station[i]]
            if j == 0:
                ax.set_title(list_station[i])
            plot_contribPie(ax, station, labels=None)
            if i == 0:
                ax.set_ylabel(list_OPtype[j], {'size': '16'} )
                ax.yaxis.labelpad = 60
    plt.subplots_adjust(top=0.95, bottom=0.16, left=0.07, right=0.93)

    if saveFig:
        plot_save("contribAllSites", OUTPUT_DIR, fmt=fmt_save)

    # ========== PLOT COEFFICIENT =================================================
    f, axes = plt.subplots(nrows=len(list_OPtype),ncols=1,sharex=True,figsize=(17,8))
    for j, OPtype in enumerate(list_OPtype):
        plot_all_coeff(list_station, OPtype, SAVE_DIR, axes[j])
    plt.legend(loc="center", bbox_to_anchor=(0.5,-0.15*len(list_OPtype)),
                   ncol=len(list_station))
    plt.subplots_adjust(top=0.95, bottom=0.16, left=0.07, right=0.93)
    if fromSource:
        xl = axes[j].get_xticklabels()
        l = list()
        for xli in xl:
            l.append(xli.get_text())
        l = [l.replace("_"," ") for l in l]
        axes[j].set_xticklabels(l, rotation=0)

    if saveFig:
        plot_save("coeffAllsites", OUTPUT_DIR, fmt=fmt_save)

    # ========== COMPARE CHEM - OP PIE CHART ========================================
    f,axes = plt.subplots(nrows=len(list_OPtype)+1,ncols=len(list_station),figsize=(17,8))
    src=[]
    h=[]
    for j, row in enumerate(axes):
        for i, ax in enumerate(row):
            if j==0:
                ax.set_title(list_station[i])
                df = sto[list_station[i%len(list_station)]].SRC
            else:
                df = sto[list_station[i%len(list_station)]].SRC \
                    * sto[list_station[i%len(list_station)]].OPi[list_OPtype[j-1]]
            src = src + [a for a in df.columns if a not in src]
            plot_contribPie(ax, df, labels=None)
            if i == 0:
                if j == 0:
                    ax.set_ylabel("Mass", {'size': '16'} )
                else:
                    ax.set_ylabel(list_OPtype[j-1], {'size': '16'} )
                ax.yaxis.labelpad = 50
    plt.subplots_adjust(top=0.95, bottom=0.16, left=0.07, right=0.93)

    if saveFig:
        plot_save("compareCHEM-OP", OUTPUT_DIR, fmt=fmt_save)



    # ========== PLOT SEASONAL CONTRIBUTION ============================
    for name in list_station:             
        f, axes = plt.subplots(nrows=1, ncols=len(list_OPtype)+2,
                               figsize=(12,3),
                               sharey=True)
        for i, plot in enumerate(["CHEM","DTTv","AAv"]):
            if i ==0:
                plot_seasonal_contribution(station, OPtype="Mass",
                                           CHEMorOP="CHEM",ax=axes[i])    
                axes[i].set_ylabel("Normalized contribution")
                axes[i].legend("")
            else:
                plot_seasonal_contribution(station, OPtype=plot,
                                           CHEMorOP="OP",ax=axes[i])    
            if i==2:
                axes[i].legend("")
            axes[i].set_xlabel(" ")

        plot_anual_contribution(station, ax=axes[-1])
        axes[i].set_xlabel(" ")

        if saveFig:
            plot_save("Normalized_contribution_"+name, OUTPUT_DIR, fmt=["png","pdf"])
