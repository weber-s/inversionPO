# -*-coding:Utf-8 -*
import sys, os
import datetime as dt
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import linalg
from scipy import polyfit
import math
import pandas as pd

def loadData(station_list, POtype_list, BasefileConc, BasefileUnc):
    conc  = {}
    unc   = {}
    for name in station_list:
        fileConc= os.path.join(fileDirPO,name+"/"+name+BasefileConc)
        fileUnc = os.path.join(fileDirPO,name+"/"+name+BasefileUnc)
        try:
            conc[name]  = pd.read_csv(fileConc,
                                      index_col=0,
                                      parse_dates=["date"],
                                      dayfirst=True)
            POtype_list.append("date")
            unc[name]   = pd.read_csv(fileUnc,
                                      index_col="date",
                                      parse_dates=["date"],
                                      dayfirst=True,
                                      usecols=POtype_list)
            POtype_list.remove("date")
        except FileNotFoundError as e:
            print("ERROR {station}: {error}.".format(error=str(e), station=name))
            print("Aborting...")
            sys.exit()
    return (conc, unc)


def to_date(strdate):
    # "31/10/2010 22:34"
    date=list()
    #print(strdate)
    for d in strdate:
        if d=='':
            return float(-999)
        date.append(dt.datetime.strptime(d, '%d/%m/%Y'))
    date = np.array(date)
    return(date)

def sourcesColor():
    color ={
        "Vehicular": '#000000',
        "Bio. burning": '#92d050',
        "Sulfate-rich": '#ff2a2a',
        "Nitrate-rich": '#ff7f2a',
        "Secondary bio": '#8c564b',
        "Sea/road salt": '#00b0f0',
        "Primary bio": '#ffc000',
        "Mineral dust": '#e9ddaf',
        "AOS/dust": '#e9ddaf',
        "Industrial": '#7030a0',
        "Débris végétaux": '#2aff80',
        "Chlorure": '#80e5ff'
    }
    color = pd.DataFrame(index=["color"], data=color)
    return color

def solveLinear(G,d,C):
    """
    Solve the linear system Gm=d with a covariance matrix of the parameters C
    with the folowing fomula:
    m = [Gt*cov(d)^-1*G]^-1*Gt*cov(d)^-1*d

    Return:
        m:      The parameters
        Res:    Resolution matrix
        Covm:   The covariance matrix of the parameters
    """
    Gt          = G.T
    invC        = linalg.inv(C)
    GtinvC      = np.dot(Gt,invC)
    invGtinvCG  = linalg.inv(np.dot(GtinvC,G))
    invGtinvCGGt= np.dot(invGtinvCG,Gt)
    Gg          = np.dot(invGtinvCGGt,invC)
    #GtGinvGt=np.dot(linalg.inv(GtG),G.T)
    #r=np.dot(GtGinvGt,b)
    Covm    = np.dot(Gg.dot(C),Gg.T)
    Res     = Gg.dot(G)
    m       = Gg.dot(d)
    return m, Covm, Res

def loadCHEM(conc, fromSource, PMother=False, factor2regroup=[]):
    """
    Select the concentration's columns to use.
     * If `fromSource=True`, then assume the concentration comes from source
    apportionment method with column=factor.
        `PMother` (default None) allows to group some factors into a single one.
        In this case, `factor2regroup` (array) should be provide with each
        element is a factor name to regroup (ex: ["Secondary bio", 1,
        "Chlorure"]
     * Otherwise, assume that the concentration is for chemical species. Only a
    subset of them are used.
    """
    if fromSource:
        #exclude = ["POAAm3","POAAµg","PODTTm3","PODTTµg"]
        #CHEM    = conc.drop(exclude, axis=1, errors="ignore")
        CHEM    = conc
        if PMother:
            try:
                All     = ["Sea/road salt","Secondary bio","Primary bio",\
                           "Mineral dust","Débris végétaux","Chlorure","AOS/dust"]
                Other   = CHEM.ix[:,All]
            except KeyError:
                pass
            CHEM["PM other"] = Other.sum(axis=1)
            CHEM.drop(All, axis=1, errors="ignore", inplace=True)
    else:
        colOK   = ("OC","EC",\
                   "Na+","NH4+","K+","Mg2+","Ca2+","MSA","Cl-","NO3-","SO42-","Oxalate",\
                   "Levoglucosan","Levoglusan","Polyols","ΣPolyols",\
                   "As","Ba","Cd","Cu","Hg","Mn","Mo","Ni","Pb","Sb","Sn","Ti","Zn","Al","Fe","Ag"\
                   "ΣHAP","ΣHOP","Σmethoxy_part")
        CHEM    = conc.ix[:, colOK]#.sort_index(axis=1)
        CHEM.dropna(axis=1, how="all", inplace=True)
    return CHEM

def reversePO(conc, unc, PO, fromSource):
    #date=to_date(conc.index)
    # ===== CHEM PART
    #CHEM    = loadCHEM(conc, fromSource)
    #label   = CHEM.columns
    #CHEM    = CHEM.as_matrix()
    # ===== PO PART
    #PO      = conc[POtype]#+conc["POAAµg"]
    #PO      = PO.as_matrix()
    #POunc   = POunc.as_matrix()
    # ===== Ensure no NaN + build covariance matrix
    #idok    = ~np.isnan(PO)
    #G       = CHEM[idok,:]
    #d       = PO[idok]
    #Covd    = np.diag(np.power(POunc[idok],2))
    # ===== Linear inversion
    #m, Covm, Resm= solveLinear(G,d,Covd)
    CHEM    = loadCHEM(conc, fromSource)
    POunc   = unc[POtype]
    rowOkV  = CHEM.T.notnull().all()
    rowOkPO = POunc.T.notnull()
    rowOk   = rowOkV & rowOkPO
    colKo   = ["POAAm3","POAAµg","PODTTm3","PODTTµg","POPerCent"]
    G       = CHEM.ix[rowOk,:].drop(colKo,
                                    axis=1,
                                    errors="ignore")
    d       = CHEM.ix[rowOk,POtype]
    Covd    = np.diag(np.power(POunc[rowOk],2))
    m, Covm, Resm= solveLinear(G=G.as_matrix(),
                               d=d.as_matrix(),
                               C=Covd)
    m = pd.DataFrame(index=CHEM.drop(colKo,axis=1,errors="ignore").columns, columns=[name], data=m)
    # ===== model reconstruction
    model       = CHEM.drop(colKo,axis=1,errors="ignore").dot(m)
    residu      = (d - model.ix[rowOk,0]).T.dot( d - model.ix[rowOk,0])
    p   = polyfit(d,model.ix[rowOk,:],1)
    p.shape = (2,)
    r2  = pd.concat([d,model],axis=1).corr()
    # ===== Save result in a dictionary
    station             = {}
    station[POtype]     = conc[POtype]
    station['CHEM']     = CHEM
    station['m']        = m
    station['G']        = G
    station['model']    = model
    station['d']        = d
    station['r2']       = r2
    station['p']        = p
    station['Covm']     = Covm
    station['Resm']     = Resm
    station['residu']   = residu
    return station



def plot_coeff(station, POtype_list, fromSource=True, saveFig=False):
    """Plot a bar plot of the intrinsique PO of the sources for all the station"""
    f,ax=plt.subplots(nrows=len(POtype_list),ncols=1,sharex=True,figsize=(17,8))
    for i, POtype in enumerate(POtype_list):
        dfVal = pd.DataFrame()
        dfUnc = pd.DataFrame()
        for name in station.keys():
            mValtmp = pd.DataFrame(index=station[name][POtype]["G"].columns,
                                   data=station[name][POtype]["m"],
                                   columns=[name])
            mUnctmp = pd.DataFrame(index=station[name][POtype]["G"].columns,
                                   data=np.sqrt(np.diag(station[name][POtype]['Covm'])),
                                   columns=[name])
            dfVal = pd.merge(dfVal, mValtmp, right_index=True, left_index=True,
                               how="outer")
            dfUnc  = pd.merge(dfUnc, mUnctmp, right_index=True, left_index=True,
                               how="outer")

        dfVal.plot(kind="bar", yerr=dfUnc, ax=ax[i], legend=False)

        ax[i].set_title(POtype)
        ax[i].set_ylabel("nmol/min/µg")
        plt.subplots_adjust(top=0.95, bottom=0.16, left=0.07, right=0.93)
        plt.legend(loc="center", bbox_to_anchor=(0.5,-0.3), ncol=len(station_list))
        if fromSource:
            l   = ax[-1].get_xticklabels()
            ax[i].set_xticklabels(l, rotation=0)
            if saveFig:
                plt.savefig("figures/coeffAllSites.png")
                plt.savefig("figures/svg/coeffAllSites.svg")
                plt.savefig("figures/pdf/coeffAllSites.pdf")

def plot_contribPie(station, POtype_list, saveFig=False):
    """Plot contribution of the PO in a Pie chart"""
    f,ax=plt.subplots(nrows=len(POtype_list),ncols=len(station_list),figsize=(17,8))
    for j,name in enumerate(station_list):
        for i, POtype in enumerate(POtype_list):
            param   = station[name][POtype]["m"]
            conc    = station[name][POtype]["G"]
            df      = conc*param.T.iloc[0]
            if (np.sum(df.dropna(axis=1))<0).any():
                continue
            ax[i][j].set_aspect('equal')
            df.sort_index(axis=1,inplace=True)
            c = sourcesColor()
            cols = c.ix["color",df.columns].values
            np.sum(df.dropna(axis=1)).plot.pie(ax=ax[i][j],
                                               shadow=False,
                                               startangle=90,
                                               colors=cols)
            ax[i][j].set_title(name)
            ax[i][j].set_ylabel("")
        plt.subplots_adjust(top=0.95, bottom=0.16, left=0.07, right=0.93)
    if saveFig:
        plt.savefig("figures/coeffAllSites.png")
        plt.savefig("figures/svg/coeffAllSites.svg")
        plt.savefig("figures/pdf/coeffAllSites.pdf")


mpl.rcParams.update({'font.size': 12})
# ========= PARAMETERS =======================================================
fileDirPO   = "/home/samuel/Documents/IGE/BdD_PO/"
#fileDirPO   = "/home/samuel/Documents/IGE/PMF/DataPMF/"


fromSource  = True


station_list= ("Frenes","Passy","Marnaz","Chamonix")
POtype_list = ["PODTTm3","POAAm3"]#PerCent"]

plotBool    = True 
saveFig     = True 

if fromSource:
    BasefileConc= "ContributionsMass.csv"
    BasefileUnc = "PO_unc.csv"
else:
    BasefileConc= "PO+CHEM_conc.csv"
    BasefileUnc = "PO+CHEM_unc.csv"

station     = {}
stoModel    = {}


# ========= LOAD DATA ========================================================
concentrations, uncertainties = loadData(station_list, POtype_list,
                                         BasefileConc, BasefileUnc)

# ========= INVERSION ========================================================
for name in station_list:
    station[name]   = {}
    for POtype in POtype_list:
        print("\n===== {name} for {PO} =====".format(name=name, PO=POtype))
        station[name][POtype] =  reversePO(concentrations[name],
                                           uncertainties[name],
                                           POtype,
                                           fromSource)
        # plot part ==================================================================

        print("\nResolution param:")
        print(np.array_str(station[name][POtype]['Resm'],suppress_small=True))
        #for i, x in enumerate(m):
            #    print("%s \t%0.4f\t± %0.4f" % (concentrations[name].keys()[i], x,
            #                                    np.sqrt(concentrations[name].Covm[i,i])))

        if plotBool:
            CHEM= station[name][POtype]["CHEM"]
            PO  = station[name][POtype][POtype]
            model = station[name][POtype]["model"]
            m   = station[name][POtype]["m"]
            p   = station[name][POtype]["p"]
            r2  = station[name][POtype]["r2"].as_matrix()
            Covm= station[name][POtype]["Covm"]
            plt.figure(figsize=(17,8))
            # time serie reconstruction/observation
            ax=plt.subplot(2,3,(1,3))
            PO.plot(ax=ax, style="b-o", label="Obs.")
            model.name = "recons."
            model.plot(ax=ax, style="r-*", label="recons.", rot=0)
            plt.title("{station} {PO}".format(station=name, PO=POtype))
            l=ax.legend(('Obs.','Recons.'))
            l.draw_frame(False)
            # scatter plot reconstruction/observation
            ax=plt.subplot(2,3,4)
            pd.concat((PO,model), axis=1).plot(ax=ax,x=[0], y=[1], kind="scatter")
            plt.plot([0,PO.max()],[0, PO.max()], '--', label="y=x")
            plt.plot([0,PO.max()],[p[1], p[0]*PO.max()+p[1]], label="linear fit")
            posy = 0.7*plt.ylim()[1]
            plt.text(0,posy,"y=%.2fx+%0.2f\nr²=%0.2f" % (p[0],p[1],r2[0,1]))
            plt.xlabel("PO obs.")
            plt.ylabel("PO reconstruct")
            plt.title("obs. vs reconstruction")
            ax.set_aspect(1./ax.get_data_ratio())
            l=plt.legend(loc="lower right")
            l.draw_frame(False)
            # factors contribution
            ax=plt.subplot(2,3,5)
            m.plot(ax=ax,
                   kind="bar",
                   yerr=np.sqrt(np.diag(Covm)),
                   ecolor="k",
                   align='center',
                   rot=-60,
                   legend=False)
            plt.ylabel("PO [nmol/min/µg]")
            #plt.ylim((-0.1,1.4))
            # Pie chart
            ax=plt.subplot(2,3,6)
            ax.set_aspect('equal')
            df = CHEM.ix[:,m.index] * m.T.iloc[0]
            df.sort_index(axis=1,inplace=True)
            c = sourcesColor()
            cols = c.ix["color",df.columns].values
            if (df.sum()>0).all():
                df.sum().plot.pie(shadow=False, startangle=90, colors=cols)
                ax.set_ylabel("")

            plt.subplots_adjust(top=0.95, bottom=0.16, left=0.07, right=0.93)

            if saveFig:
                plt.savefig("figures/svg/inversion"+name+POtype+".svg")
                plt.savefig("figures/pdf/inversion"+name+POtype+".pdf")
                plt.savefig("figures/inversion"+name+POtype+".png")

            #plt.tight_layout(h_pad=0.4)

# ========== PLOT COEFFICIENT =================================================
plot_coeff(station, POtype_list, fromSource=fromSource, saveFig=saveFig)


# ========== CONTRIBUTION PIE CHART ===========================================
plot_contribPie(station, POtype_list, saveFig=saveFig)






