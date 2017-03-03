import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import sklearn
from scipy import linalg
from scipy import polyfit
from sklearn import linear_model
from plot_utility import *

class Station:
    def __init__(self,name=None,CHEM=None,PO=None,m=None,hasPO=True):
        self.name   = name
        self.CHEM   = CHEM
        if hasPO:
            self.PO     = PO
            self.m      = m
            self.model  = pd.Series((CHEM*m).sum(axis=1), name=name)
            if (m>=0).all():
                self.pie    = pd.Series((CHEM*m).sum(), name=name)
            else:
                self.pie=pd.Series(index=CHEM.columns)
            self.p      = polyfit(PO,self.model,1)
            self.p.shape= (2,)
            self.r2     = pd.concat([PO,self.model],axis=1).corr().as_matrix()
        else:
            self.PO     = None
            self.m      = pd.Series(index=CHEM.columns)
            self.model  = pd.Series(index=CHEM.index)
            self.pie    = pd.Series(index=CHEM.columns)
            self.p      = None
            self.r2     = None
        self.hasPO  = hasPO
        self.pie.sort_index(inplace=True)
        self.m.sort_index(inplace=True)


def lsqr(CHEM, POunc, PO, fromSource):
    G   = CHEM.as_matrix()
    d   = PO.as_matrix()
    C   = np.diag(np.power(POunc.as_matrix(),2))

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

    m = pd.Series(index=CHEM.columns, data=m)
    return m, Covm, Res

def solve_inversion(d, G, std, x_min=None, x_max=None):
    x   = pulp.LpVariable.dicts("PO", G.columns, x_min, x_max)
    m   = pulp.LpVariable("to_minimize", 0)
    lp_prob = pulp.LpProblem("Minmax Problem", pulp.LpMinimize)
    lp_prob += m, "Minimize_the_maximum"

    for i in range(len(G.index)):
        label = "Val %d" % i
        label2 = "Val2 %d" % i
        dot_G_x = pulp.lpSum([G.ix[i][j] * x[j] for j in G.columns])
        condition = (d[i] - dot_G_x) <= m + 0.5*std[i]
        lp_prob += condition, label
        condition = (d[i] - dot_G_x) >= -m - 0.5*std[i]
        lp_prob += condition, label2
    lp_prob.solve()

    return lp_prob

INPUT_DIR = "/home/samuel/Documents/IGE/BdD_PO/"
OUTPUT_DIR= "/home/samuel/Documents/IGE/inversionPO/figures/inversionLARS/"
list_station= ["Nice","Marnaz","Passy","Chamonix"]
#list_station= ("Passy",)
list_POtype = ["PODTTm3","POAAm3"]


OrdinaryLeastSquare = False
MachineLearning     = not(OrdinaryLeastSquare)

fromSource  = True
saveFig     = False
plotTS      = True

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
for POtype in list_POtype:
    sto[POtype]=dict()
    print("=============="+POtype+"====================")
    s = pd.Series()
    pie = pd.Series()
    for name in list_station:
        print("=============="+name+"====================")
        CHEM    = pd.read_csv(INPUT_DIR+name+"/"+name+name_File, index_col="date", parse_dates=["date"], dayfirst=True)
        PO      = pd.read_csv(INPUT_DIR+name+"/"+name+"PO.csv", index_col="date", parse_dates=["date"], dayfirst=True)
        if not(fromSource):
            # select the species we want
            colOK   = ("OC","EC",\
                       "Na+","NH4+","K+","Mg2+","Ca2+","MSA","Cl-","NO3-","SO42-","Oxalate",\
                       "Levoglucosan","Levoglusan","Polyols","ΣPolyols",\
                       "As","Ba","Cd","Cu","Hg","Mn","Mo","Ni","Pb","Sb","Sn","Ti","Zn","Al","Fe","Ag"\
                       "ΣHAP","ΣHOP","Σmethoxy_part")
            CHEM    = CHEM.ix[:,colOK]
            CHEM.dropna(axis=1, how="all", inplace=True)

        if not(POtype in PO.columns):
            sto[POtype][name] = Station(name=name, CHEM=CHEM, hasPO=False)
            stmp= pd.Series(sto[POtype][name].m,name=name)
            s   = pd.concat([s, stmp],axis=1)
            pie = pd.concat([pie, sto[POtype][name].m],axis=1)
            continue
        POunc   = PO["unc"+POtype]
        PO      = PO[POtype]

        TMP     = pd.concat([PO,POunc,CHEM],axis=1,join="inner")
        TMP.dropna(inplace=True)

        PO      = TMP[POtype]
        POunc   = TMP["unc"+POtype]
        CHEM    = TMP.ix[:,2:]

        # Machine learning from scikit
        if MachineLearning:
            # ==== Separate the dataset into a training and test set
            X       = CHEM.values
            y       = PO.values
            #X_train = X[indices[:-nb]][:]
            #np.random.seed(0)
            #indices = np.random.permutation(len(PO))

            #nb      = 1
            #X_test  = X[indices[-nb:]][:]
            #y_train = y[indices[:-nb]]
            #y_test  = y[indices[-nb:]]

            #regr = linear_model.LinearRegression()
            #regr = linear_model.Lasso(positive=True)
            #regr = linear_model.ElasticNet(l1_ratio=0, positive=True)
            #regr = linear_model.Ridge(alpha=0.01)
            regr = linear_model.Lars(positive=True)
            regr.fit(X, y)

            tmp = dict()
            for i, v in enumerate(regr.coef_):
                tmp[CHEM.columns[i]] = v
                print(CHEM.columns[i], "=", v)
            m = pd.Series(tmp,name=name)
            #newname = renameIndex()
            #tmp.rename(newname, inplace=True)

        elif OrdinaryLeastSquare:
            # Ordinary least square method (implemeted by myself)
            m, Covm, Res =  lsqr(CHEM,
                                 POunc,
                                 PO,
                                 fromSource)
            m.name  = name

        s   = pd.concat([s,m],axis=1)
        sto[POtype][name] = Station(name=name,
                                    CHEM=CHEM,
                                    PO=PO,
                                    m=m)
        if plotTS:
            plot_station(sto[POtype][name],POtype)
            if saveFig:
                plt.savefig(OUTPUT_DIR+"svg/inversion"+name+POtype+".svg")
                plt.savefig(OUTPUT_DIR+"pdf/inversion"+name+POtype+".pdf")
                plt.savefig(OUTPUT_DIR+"inversion"+name+POtype+".png") 

    saveCoeff[POtype] = s.ix[:,list_station]

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
            ax.yaxis.labelpad = 50
plt.subplots_adjust(top=0.95, bottom=0.16, left=0.07, right=0.93)

if saveFig:
    plt.savefig(OUTPUT_DIR+"contribAllSites.png")
    plt.savefig(OUTPUT_DIR+"svg/contribAllSites.svg")
    plt.savefig(OUTPUT_DIR+"pdf/contribAllSites.pdf")

# ========== PLOT COEFFICIENT =================================================
f, axes = plt.subplots(nrows=len(list_POtype),ncols=1,sharex=True,figsize=(17,8))
for j, ax in enumerate(axes):
    station = saveCoeff[list_POtype[j]]
    plot_coeff(station, ax=ax)
    ax.set_title(list_POtype[j])
    ax.set_ylabel("nmol/min/µg")
plt.legend(loc="center", bbox_to_anchor=(0.5,-0.1*len(list_POtype)),
           ncol=len(list_station))
plt.subplots_adjust(top=0.95, bottom=0.16, left=0.07, right=0.93)
if fromSource:
    l   = ax.get_xticklabels() # -2 because ax[-1] is ""
    ax.set_xticklabels(l, rotation=0)

if saveFig:
    plt.savefig(OUTPUT_DIR+"coeffAllSites.png")
    plt.savefig(OUTPUT_DIR+"svg/coeffAllSites.svg")
    plt.savefig(OUTPUT_DIR+"pdf/coeffAllSites.pdf")


