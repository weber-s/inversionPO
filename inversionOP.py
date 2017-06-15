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



INPUT_DIR = "/home/samuel/Documents/IGE/BdD/BdD_OP/"

# list_station= ["Nice","Frenes","Passy","Chamonix", "Marnaz"]
list_station= ["ANDRA","Nice","Frenes","Passy","Chamonix",
               "Marnaz","Marseille","PdB"]
# list_station= ["Marseille","Frenes","Nice","PdB"]

# list_OPtype = ["AAv","DTTv","DCFHv"]
list_OPtype = ["AAv","DTTv"]

# format to save plot
fmt_save    =["png","pdf","svg"]


plt.interactive(False)

# Choose the inversion method (could be OLS, WLS, GLS or ML)
inversion_method = "WLS"
OUTPUT_DIR="/home/samuel/Documents/IGE/inversionOP/figures/inversion"+inversion_method+"_wo_outliers/"
SAVE_DIR="/home/samuel/Documents/IGE/inversionOP/results/inversion"+inversion_method+"_wo_outliers/"

fromSource  = True
saveFig     = False
plotTS      = False
plotBar     = False
plotRegplot = False
saveResult  = False
sum_sources = False
plotAll     = False

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

        if not(fromSource):
            # select the species we want
            colOK   = ("OC","EC",\
                       "Na+","NH4+","K+","Mg2+","Ca2+","MSA","Cl-","NO3-","SO42-","Oxalate",\
                       "Levoglucosan","Levoglusan","Polyols","ΣPolyols",\
                       "As","Ba","Cd","Cu","Hg","Mn","Mo","Ni","Pb","Sb","Sn","Ti","Zn","Al","Fe","Ag"\
                       "ΣHAP","ΣHOP","Σmethoxy_part")
            CHEM    = CHEM.ix[:,colOK]
            CHEM.dropna(axis=1, how="all", inplace=True)
        else:
            # rename columns
            CHEM = setSourcesCategories(CHEM)
            CHEM.sort_index(axis=1, inplace=True)
            if sum_sources:
                # print(CHEM)
                to_merge =["Bio_burning", "Bio_burning1","Bio_burning2",\
                          "Vehicular", "Vehicular_ind","Vehicular_dir"]
                for i in range(0,int(len(to_merge)/3)+3,3):
                    try:
                        CHEM[to_merge[i]] = CHEM[to_merge[i+1:i+3]].sum(axis=1)
                        CHEM.drop(to_merge[i+1:i+3], axis=1,inplace=True)
                    except:
                        pass
                        # print("The sources {merge} are not in the site {site}".format(merge=to_merge[i+1:i+3], site=name))

            # CHEM = CHEM.div(CHEM.sum(axis=1),axis="index")

        if not(OPtype in OP.columns) or OP[OPtype].isnull().all():
            sto[OPtype][name] = Station(name=name, CHEM=CHEM, hasOP=False)
            pie = pd.concat([pie, sto[OPtype][name].m],axis=1)
            continue
        if (inversion_method in "OLS|WLS|GLS") and OP["SD_"+OPtype].isnull().all():
            sto[OPtype][name] = Station(name=name, CHEM=CHEM, hasOP=False)
            pie     = pd.concat([pie, sto[OPtype][name].m],axis=1)
            continue

        # ==== Drop days with missing values
        TMP = pd.DataFrame.join(CHEM,OP[[OPtype,"SD_"+OPtype]],how="inner")
        TMP.dropna(inplace=True)
        OP   = TMP[OPtype]
        OPunc= TMP["SD_"+OPtype]
        CHEM = TMP[CHEM.columns]
        OP.name = OPunc.name = CHEM.name = name
        

        # ==== Different inversion method
        if inversion_method == "ML":
            # Machine learning from scikit
            m       = solve_scikit_linear_regression(X=CHEM.values, y=OP.values,
                                                    index=CHEM.columns)
            m.name  = name
            covm    = pd.Series(index=CHEM.columns)
            yerr    = None
        elif inversion_method == "GLS":
            regr    = solve_GLS(X=CHEM, y=OP, sigma=1/OPunc**2)
            # print(regr.summary())

            m = pd.Series(index=CHEM.columns, data=0)
            m[regr.params.index]= regr.params
            m.name = name
            covm = pd.Series(index=CHEM.columns, data=0)
            covm[regr.params.index] = regr.bse
            covm.name = name
            yerr    = None

        elif inversion_method == "WLS":
            regr    = solve_WLS(X=CHEM, y=OP, sigma=1/OPunc**2)
            # print(regr.summary())
         
            m = pd.Series(index=CHEM.columns, data=0)
            m[regr.params.index]= regr.params
            m.name = name
            covm = pd.Series(index=CHEM.columns, data=0)
            covm[regr.params.index] = regr.bse
            covm.name = name

            yerr, iv_l, iv_l = wls_prediction_std(regr)

        elif inversion_method == "OLS":
            # Ordinary least square method (implemeted by myself)
            m, Covm, Res =  solve_lsqr(G=CHEM.as_matrix(),
                                       d=OP.as_matrix(),
                                       Covd=np.diag(np.power(OPunc.as_matrix(),2)))
            m       = pd.Series(index=CHEM.columns, data=m)
            m.name  = name
            covm    = pd.Series(index=m.index,data=np.sqrt(np.diag(Covm)))
            covm.name = name
            yerr    = None
        else:
            print("Choose an inversion method")
        
        # ==== Store the result
        sto[OPtype][name] = Station(name=name,
                                    CHEM=CHEM, OP=OP, OPunc=OPunc,
                                    m=m, covm=covm, yerr=yerr, reg=regr)

        if plotTS:
            plot_station(sto[OPtype][name],OPtype)
            if saveFig:
                plot_save("inversion"+name+OPtype, OUTPUT_DIR, fmt=fmt_save)
        if plotBar:
            plot_ts_reconstruction_OP(sto[OPtype][name],OPtype)
            if saveFig:
                plot_save("reconstructionPerSource_"+name+OPtype, OUTPUT_DIR,
                          fmt=fmt_save)


        if saveResult:
            result2csv(sto[OPtype][name],saveDir=SAVE_DIR,OPtype=OPtype)

            with open(SAVE_DIR+"/"+name+OPtype+".pickle", "wb") as handle:
                pickle.dump(sto[OPtype][name], handle, protocol=pickle.HIGHEST_PROTOCOL)
        plt.close("all")

saveCoeff = dict()
saveCovm = dict()
for OPtype in list_OPtype:
    saveCoeff[OPtype] = pd.concat([sto[OPtype][name].m for name in list_station], axis=1)
    saveCovm[OPtype] = pd.concat([sto[OPtype][name].covm for name in list_station], axis=1)

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
            plot_contribPie(ax, station)
            if i == 0:
                ax.set_ylabel(list_OPtype[j], {'size': '16'} )
                ax.yaxis.labelpad = 60
    plt.subplots_adjust(top=0.95, bottom=0.16, left=0.07, right=0.93)

    if saveFig:
        plot_save("contribAllSites", OUTPUT_DIR, fmt=fmt_save)

    # ========== PLOT COEFFICIENT =================================================
    f, axes = plt.subplots(nrows=len(list_OPtype),ncols=1,sharex=True,figsize=(17,8))
    for j, OP_type in enumerate(list_OPtype):
        plot_all_coeff(list_station, OP_type, SAVE_DIR, axes[j])
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
    for j, row in enumerate(axes):
        for i, ax in enumerate(row):
            if j==0:
                ax.set_title(list_station[i])
                df = sto[list_OPtype[j]][list_station[i]].pieCHEM
            else:
                df = sto[list_OPtype[j-1]][list_station[i]].pie
            plot_contribPie(ax, df)
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
        f, axes = plt.subplots(nrows=1, ncols=len(list_OPtype)+1,
                               figsize=(12,3),
                               sharey=True)
        for i, plot in enumerate(["CHEM","DTTv","AAv"]):
            if i ==0:
                plot_seasonal_contribution(sto["DTTv"][name], OPtype="Mass",
                                           CHEMorOP="CHEM",ax=axes[i])    
                axes[i].set_ylabel("Normalized contribution")
                axes[i].legend("")
            else:
                plot_seasonal_contribution(sto[plot][name], OPtype=plot,
                                           CHEMorOP="OP",ax=axes[i])    
            # if i==2:
            axes[i].legend("")

            axes[i].set_xlabel(" ")
        plot_save("Normalized_contribution_"+name, OUTPUT_DIR, fmt=["png","pdf"])
