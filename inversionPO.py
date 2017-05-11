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



INPUT_DIR = "/home/samuel/Documents/IGE/BdD/BdD_PO/"
OUTPUT_DIR= "/home/samuel/Documents/IGE/inversionPO/figures/inversionLARS/"
SAVE_DIR  = "/home/samuel/Documents/IGE/inversionPO/results/inversionLARS/"
list_station= ["Nice","Frenes","Passy","Chamonix", "Marnaz"]
# list_station= ["Passy"]

list_POtype = ["DTTv","AAv"]

# plt.interactive(True)

OrdinaryLeastSquare     = False
GeneralizedLeastSquare  = False
MachineLearning         = True 

fromSource  = True
saveFig     = True
plotTS      = True
plotBar     = True
saveResult  = True
sum_sources = True
plotAll     = True

if fromSource:
    name_File="_ContributionsMass_positive.csv"
else:
    name_File="CHEM_conc.csv"


# sort list in order to always have the same order
list_station.sort()
list_POtype.sort()
# initialize stuff
sto = dict()
saveCoeff = dict()
saveCovm = dict()
pvalues = dict()
for POtype in list_POtype:
    sto[POtype]=dict()
    print("=============="+POtype+"====================")
    s = pd.Series()
    cov_all = pd.Series()
    pie = pd.Series()
    for name in list_station:
        print("=============="+name+"====================")
        CHEM = load_CHEMorPO(name, INPUT_DIR, CHEMorPO="CHEM", fromSource=fromSource)
        PO   = load_CHEMorPO(name, INPUT_DIR, CHEMorPO="PO")

        if not(fromSource):
            # select the species we want
            colOK   = ("OC","EC",\
                       "Na+","NH4+","K+","Mg2+","Ca2+","MSA","Cl-","NO3-","SO42-","Oxalate",\
                       "Levoglucosan","Levoglusan","Polyols","ΣPolyols",\
                       "As","Ba","Cd","Cu","Hg","Mn","Mo","Ni","Pb","Sb","Sn","Ti","Zn","Al","Fe","Ag"\
                       "ΣHAP","ΣHOP","Σmethoxy_part")
            CHEM    = CHEM.ix[:,colOK]
            CHEM.dropna(axis=1, how="all", inplace=True)
        
        # rename columns
        CHEM = setSourcesCategories(CHEM)

        if sum_sources:
            # print(CHEM)
            to_mergeBB = ["Bio_burning1","Bio_burning2"]
            to_mergeVeh= ["Vehicular_ind","Vehicular_dir"]
            try:
                CHEM["Bio_burning"] = CHEM[to_mergeBB].sum(axis=1)
                CHEM.drop(to_mergeBB, axis=1,inplace=True)
            except:
                print("The sources {merge} are not in the site {site}".format(merge=to_mergeBB, site=name))
            try:
                CHEM["Vehicular"] = CHEM[to_mergeVeh].sum(axis=1)
                CHEM.drop(to_mergeVeh, axis=1,inplace=True)
            except:
                print("The sources {merge} are not in the site {site}".format(merge=to_mergeVeh, site=name))

        if not(POtype in PO.columns) or PO[POtype].isnull().all():
            sto[POtype][name] = Station(name=name, CHEM=CHEM, hasPO=False)
            stmp= pd.Series(sto[POtype][name].m,name=name)
            covtmp = pd.Series(sto[POtype][name].covm,name=name) 
            s   = pd.concat([s, stmp],axis=1)
            cov_all = pd.concat([cov_all, covtmp],axis=1)
            pie = pd.concat([pie, sto[POtype][name].m],axis=1)
            continue
        if GeneralizedLeastSquare and PO["SD_"+POtype].isnull().all():
            sto[POtype][name] = Station(name=name, CHEM=CHEM, hasPO=False)
            covtmp = pd.Series(sto[POtype][name].covm,name=name) 
            stmp= pd.Series(sto[POtype][name].m,name=name)
            s   = pd.concat([s, stmp],axis=1)
            cov_all = pd.concat([cov_all, covtmp],axis=1)
            pie = pd.concat([pie, sto[POtype][name].m],axis=1)
            continue

        # ==== Drop day with missing values
        TMP = pd.DataFrame.join(CHEM,PO[[POtype,"SD_"+POtype]],how="inner")
        TMP.dropna(inplace=True)
        PO   = TMP[POtype]
        POunc= TMP["SD_"+POtype]
        CHEM = TMP[CHEM.columns]
        PO.name = POunc.name = CHEM.name = name
        
        if name == "Frenes":
            pvalues[POtype]=pd.DataFrame(index=CHEM.columns)
        # ==== Different inversion method
        if MachineLearning:
            # Machine learning from scikit
            m       = solve_scikit_linear_regression(X=CHEM.values, y=PO.values,
                                                    index=CHEM.columns)
            m.name  = name
            covm    = pd.Series(index=CHEM.columns)
        elif GeneralizedLeastSquare:
            goForWLS = CHEM.copy()
            regr = sm.WLS(PO, goForWLS, sigma=POunc**2).fit()
            while True:
                regr = sm.WLS(PO, goForWLS, sigma=POunc**2).fit()
                # print(regr.summary())
                if name == "Frenes":
                    pvalues[POtype] = pd.concat([pvalues[POtype],regr.pvalues],axis=1)
                    print(regr.summary())
                # if (regr.pvalues > 0.05).any():
                if (regr.params < 0).any():
                    # Some variable are 0, drop them.
                    # goForWLS.drop(goForWLS.columns[regr.pvalues>0.05],axis=1,inplace=True)
                    # goForWLS.drop(goForWLS.columns[regr.pvalues == max(regr.pvalues)],axis=1,inplace=True)
                    goForWLS.drop(goForWLS.columns[regr.params == min(regr.params)],axis=1,inplace=True)
                else:
                    # Ok, the run converged
                    break
                if goForWLS.shape[1]==0:
                    # All variable were droped... Pb
                    print("Warning: The run did not converge...")
                    break
             
            m = pd.Series(index=CHEM.columns, data=0)
            m[goForWLS.columns]= regr.params
            m.name = name
            covm = pd.Series(index=CHEM.columns, data=0)
            covm[goForWLS.columns] = regr.bse
            covm.name = name
            # print(regr.summary())
        elif OrdinaryLeastSquare:
            # Ordinary least square method (implemeted by myself)
            m, Covm, Res =  solve_lsqr(G=CHEM.as_matrix(),
                                       d=PO.as_matrix(),
                                       Covd=np.diag(np.power(POunc.as_matrix(),2)))
            m       = pd.Series(index=CHEM.columns, data=m)
            m.name  = name
            covm    = pd.Series(index=m.index,data=np.sqrt(np.diag(Covm)))
            covm.name = name
        
        # ==== Store the result
        s   = pd.concat([s,m],axis=1)
        cov_all = pd.concat([cov_all,covm],axis=1)
        sto[POtype][name] = Station(name=name,
                                    CHEM=CHEM,
                                    PO=PO,
                                    POunc=POunc,
                                    m=m,
                                    covm=covm)
        if plotTS:
            plot_station(sto[POtype][name],POtype)
            if saveFig:
                plt.savefig(OUTPUT_DIR+"svg/inversion"+name+POtype+".svg")
                plt.savefig(OUTPUT_DIR+"pdf/inversion"+name+POtype+".pdf")
                plt.savefig(OUTPUT_DIR+"inversion"+name+POtype+".png") 
        if plotBar:
            plot_ts_reconstruction_PO(sto[POtype][name],POtype)
            if saveFig:
                plt.savefig(OUTPUT_DIR+"svg/reconstructionPerSource_"+name+POtype+".svg")
                plt.savefig(OUTPUT_DIR+"pdf/reconstructionPerSource_"+name+POtype+".pdf")
                plt.savefig(OUTPUT_DIR+"reconstructionPerSource_"+name+POtype+".png") 


        if saveResult:
            result2csv(sto[POtype][name],saveDir=SAVE_DIR,POtype=POtype)

            with open(SAVE_DIR+"/"+name+POtype+".pickle", "wb") as handle:
                pickle.dump(sto[POtype][name], handle, protocol=pickle.HIGHEST_PROTOCOL)

    saveCoeff[POtype] = s.ix[:,list_station]
    saveCovm[POtype] = cov_all.ix[:,list_station]


if plotAll:
    # ========== CONTRIBUTION PIE CHART ===========================================
    f,axes = plt.subplots(nrows=len(list_POtype),ncols=len(list_station),figsize=(17,8))
    #ax.shape = (np.sum(ax.shape),) 
    for j, row in enumerate(axes):
        for i, ax in enumerate(row):
            station=sto[list_POtype[j]][list_station[i]]
            if j == 0:
                ax.set_title(list_station[i])
            plot_contribPie(ax, station)
            if i == 0:
                ax.set_ylabel(list_POtype[j], {'size': '16'} )
                ax.yaxis.labelpad = 60
    plt.subplots_adjust(top=0.95, bottom=0.16, left=0.07, right=0.93)

    if saveFig:
        plt.savefig(OUTPUT_DIR+"contribAllSites.png")
        plt.savefig(OUTPUT_DIR+"svg/contribAllSites.svg")
        plt.savefig(OUTPUT_DIR+"pdf/contribAllSites.pdf")

    # ========== PLOT COEFFICIENT =================================================
    f, axes = plt.subplots(nrows=len(list_POtype),ncols=1,sharex=True,figsize=(17,8))
    for j, ax in enumerate(axes):
        coeff = saveCoeff[list_POtype[j]]
        covm  = saveCovm[list_POtype[j]]
        plot_coeff(coeff, yerr=covm, ax=ax)
        ax.set_title(list_POtype[j])
        ax.set_ylabel("nmol/min/µg")
        plt.legend(loc="center", bbox_to_anchor=(0.5,-0.15*len(list_POtype)),
                   ncol=len(list_station))
        plt.subplots_adjust(top=0.95, bottom=0.16, left=0.07, right=0.93)
        if fromSource:
            # l   = ax.get_xticklabels() # -2 because ax[-1] is ""
            l   = coeff.index # -2 because ax[-1] is ""
            l = [l.replace("_"," ") for l in l]
            ax.set_xticklabels(l, rotation=-10)

    if saveFig:
        plt.savefig(OUTPUT_DIR+"coeffAllSites.png")
        plt.savefig(OUTPUT_DIR+"svg/coeffAllSites.svg")
        plt.savefig(OUTPUT_DIR+"pdf/coeffAllSites.pdf")

    # ========== COMPARE CHEM - PO PIE CHART ========================================
    f,axes = plt.subplots(nrows=len(list_POtype)+1,ncols=len(list_station),figsize=(17,8))
    for j, row in enumerate(axes):
        for i, ax in enumerate(row):
            if j==0:
                ax.set_title(list_station[i])
                df = sto[list_POtype[j]][list_station[i]].pieCHEM
            else:
                df = sto[list_POtype[j-1]][list_station[i]].pie
            plot_contribPie(ax, df)
            if i == 0:
                if j == 0:
                    ax.set_ylabel("Mass", {'size': '16'} )
                else:
                    ax.set_ylabel(list_POtype[j-1], {'size': '16'} )
                ax.yaxis.labelpad = 50
    plt.subplots_adjust(top=0.95, bottom=0.16, left=0.07, right=0.93)

    if saveFig:
       plt.savefig(OUTPUT_DIR+"compareCHEM-PO_AllSites.png")
       plt.savefig(OUTPUT_DIR+"svg/compareCHEM-PO_AllSites.svg")
       plt.savefig(OUTPUT_DIR+"pdf/compareCHEM-PO_AllSites.pdf")


