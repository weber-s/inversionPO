import pandas as pd
import numpy as np
import pulp

class Station:
    def __init__(self,name=None,CHEM=None,PO=None,m=None):
        self.name   = name
        self.CHEM   = CHEM
        self.PO     = PO
        self.m      = m
        self.pie    = pd.concat([pie, pd.Series((CHEM*m).sum(), name=name)],axis=1)

def renameIndex():
    rename={
        "PO_Bio._burning": "Bio. burning",
        "PO_Industrial": "Industrial",
        "PO_Mineral_dust": "Mineral dust",
        "PO_Nitrate_rich": "Nitrate-rich",
        "PO_Primary_bio": "Primary bio",
        "PO_Sea_road_salt": "Sea/road salt",
        "PO_Secondary_bio": "Secondary bio",
        "PO_Sulfate_rich": "Sulfate-rich",
        "PO_Vehicular": "Vehicular",
        "PO_AOS_dust": "AOS/dust",
        "PO_Débris_végétaux": "Débris végétaux",
        "PO_Chlorure": "Chlorure"
        }
    return rename

def solve_inversion(d, G, std, x_min=None, x_max=None):
    x   = pulp.LpVariable.dicts("PO", G.columns, x_min, x_max)
    m   = pulp.LpVariable("to_minimize", 0)
    lp_prob = pulp.LpProblem("Minmax Problem", pulp.LpMinimize)
    lp_prob += m, "Minimize_the_maximum"

    for i in range(len(G.index)):
        label = "Val %d" % i
        label2 = "Val2 %d" % i
        dot_G_x = pulp.lpSum([G.ix[i][j] * x[j] for j in G.columns])
        condition = (d[i] - dot_G_x) <= m + std[i]
        lp_prob += condition, label
        condition = (d[i] - dot_G_x) >= -m - std[i]
        lp_prob += condition, label2
    #lp_prob.writeLP("MinmaxProblem.lp")  # optional
    lp_prob.solve()

    return lp_prob

DIR = "/home/samuel/Documents/IGE/BdD_PO/"
list_name   = ("Passy","Marnaz","Chamonix")
list_POtype = ("PODTTm3","POAAm3","POPerCent")

sto = dict()
save = dict()
savepie = dict()
for POtype in list_POtype:
    sto[POtype]=dict()
    print("=============="+POtype+"====================")
    s = pd.Series()
    pie = pd.Series()
    for name in list_name:
        print("=============="+name+"====================")
        PO      = pd.read_csv(DIR+name+"/"+name+"PO.csv", index_col="date", parse_dates=["date"], dayfirst=True)
        POunc   = PO["unc"+POtype]
        PO      = PO[POtype]
        CHEM    = pd.read_csv(DIR+name+"/"+name+"ContributionsMass.csv", index_col="date", parse_dates=["date"], dayfirst=True)   

        rowOKPO = PO.T.notnull()
        rowOkV  = CHEM.T.notnull().all()
        rowOK   = rowOKPO & rowOkV

        PO      = PO[rowOK]
        POunc   = POunc[rowOK]
        CHEM    = CHEM.ix[rowOK,:]


        lp=solve_inversion(PO.values, CHEM, POunc.values, x_min=0)
        
        tmp = dict()
        print("Status:", pulp.LpStatus[lp.status])
        for i, v in enumerate(lp.variables()):
            if i == len(lp.variables())-1:
                break
            tmp[v.name] = v.varValue
            print(v.name, "=", v.varValue)
        print("Total Cost =", pulp.value(lp.objective))
        tmp = pd.Series(tmp,name=name)
        newname = renameIndex()
        tmp.rename(newname, inplace=True)
        pie = pd.concat([pie, pd.Series((CHEM*tmp).sum(), name=name)],axis=1)
        s   = pd.concat([s,tmp],axis=1)
        sto[POtype][name] = Station(name=name,
                                    CHEM=CHEM,
                                    PO=PO,
                                    m=tmp)

    s.dropna(axis=1, how="all", inplace=True)
    pie.dropna(axis=1, how="all", inplace=True)
    save[POtype] = s
    savepie[POtype] = pie

for po in list_POtype:
    save[po].plot.bar(rot=0, title=po)
    savepie[po].plot.pie(rot=0, title=po, subplots=True, legend=None)
