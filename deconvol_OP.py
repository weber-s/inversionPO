
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# # OP apportionment
# 
# Weber S, ...

# ## Abstract
# 
# Oxidative potential (OP) has been proposed as a measure of toxicity of PM...

# ## Introduction
# 
# ...

# ## Methodology
# 
# ### Sampling sites
# 
# ### Analysis
# 
# #### Chemical
# 
# #### OP AA/DTT(/DCFH?)
# 
# ### Statistical framework
# 
# #### PMF
# Blabla
# 
# Evolution from the SOURCES programme: SOURCESv2
# 
# PMF profils:
# 
# - base run (same as SOURCES) 
# - constraints (same as SOURCES)
# - Bootstrap
# - mean of bootstrap as reference for the profils (SOURCESv2, not from the ref. constraint run)
# - PM contribution reconstructed from the mean PM of each bootstrap profils (SOURCESv2, not from the ref. constraint run) 
# 
# #### Mulitple linear regression
# 
# 
# 

# In[2]:


import pandas as pd
import statsmodels.api as sm 
from statsmodels.tools.tools import add_constant

def solve_WLS(X=None, y=None, sigma=None):
    """
    Solve a multiple linear problem using statsmodels WLS with positivity constraint
    according to Weber et al. (2018)
    """
    goForWLS = add_constant(X.copy())
    regr = sm.WLS(y, goForWLS, weights=sigma, cov_type="fixed_scale").fit()
    while True:
        regr = sm.WLS(y, goForWLS, weights=sigma, cov_type="fixed_scale").fit()
        paramstmp=regr.params.copy()
        paramstmp["const"]=10
        # if (regr.pvalues > 0.05).any():
        if (paramstmp < 0).any():
            # Some variable are 0, drop them.
            # goForWLS.drop(goForWLS.columns[regr.pvalues>0.05],axis=1,inplace=True)
            # goForWLS.drop(goForWLS.columns[regr.pvalues == max(regr.pvalues)],axis=1,inplace=True)
            goForWLS.drop(goForWLS.columns[paramstmp == min(paramstmp)],axis=1,inplace=True)
            # print(regr.summary())
        else:
            # Ok, the run converged
            break
        if goForWLS.shape[1]==0:
            # All variable were droped... Pb
            print("Warning: The run did not converge...")
            break
    # print(regr.summary())
    return regr


# ## Results

# In[34]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sqlite3
from Station import Station
plt.interactive(True)

dbPM = "/home/webersa/Documents/BdD/BdD_PM/db.sqlite"
conn = sqlite3.connect(dbPM)
dfPMF_all = pd.read_sql("SELECT * FROM PMF_contributions WHERE programme in ('SOURCES', 'SOURCES2');", conn)
dfPMF_profiles = pd.read_sql("SELECT * FROM PMF_profiles WHERE programme in ('SOURCES', 'SOURCES2');", conn)
dfPMF_all.dropna(axis=1, how="all", inplace=True)
dfPM_all = pd.read_sql("SELECT * FROM values_all;", conn)
conn.close()

list_OPtype = ["DTTv", "DCFHv", "AAv"]
list_OPtype = ["DTTv", "AAv"]
list_station = [
    #"PdB", "MRS-5av", "Nice", "Aix-en-provence",
    #"GRE-fr", "Marnaz", "Chamonix", "Passy",
    #"Nogent", "Roubaix", #"Rouen",
    #"STG-cle", #"Talence"
    "Chamonix", "GRE-fr"
]

map_station_name = {
    "CHAM": "Chamonix",
    "GRE": "GRE-fr"
}

dfPMF_all.replace(map_station_name, inplace=True)
dfPMF_profiles.replace(map_station_name, inplace=True)

#dfPMF_all["station"].unique()
NBOOT = 50

list_station.sort()
list_OPtype.sort()
sto = {}
for name in list_station:
    if name == "Rouen":
        continue
    if name not in dfPMF_all["station"].unique():
        continue

    print(name)
    station = Station(name=name, list_OPtype=list_OPtype)
    station.load_SRC(
        dfcontrib=dfPMF_all[dfPMF_all["station"]==name],
        dfprofile=dfPMF_profiles[dfPMF_profiles["station"]==name]
    )
    #print(station.SRC.columns)
    #station.SRC.rename({"HFO":"Traffi_non-exhaust"}, axis=1, inplace=True)
    station.setSourcesCategories()
    if "Lens-2011" in name:
        name = "Lens"
    station.load_OP(dfPM_all[dfPM_all["station"]==name])
    station.OPi = pd.DataFrame(index=station.SRC.columns, columns=list_OPtype)
    
    for OPtype in list_OPtype:
        if not(OPtype in station.OP.columns) or station.OP[OPtype].isnull().all() or station.OP[OPtype].empty:
            print("WARNING: no OP {}".format(OPtype))
            continue
        
        df = station.SRC.merge(station.OP[[OPtype, "SD_"+OPtype]], left_index=True, right_index=True, how="inner")
        
        # drop known extreme value
        mask_outliers = [False] * len(df.index)
        if name == "GRE-fr":
            mask_outliers = df.index != pd.to_datetime('2013-04-26')
            df = df[mask_outliers]
        
        df.dropna(inplace=True)
        if len(df)<=50:
            print("WARNING: not enought commun index: {}".format(len(df)))
            continue
        OP = df[OPtype]
        OPunc = df["SD_"+OPtype]
        SRC = df[station.SRC.columns]
        OP.name = OPunc.name = SRC.name = name
        
        # solve WLS
        regr = solve_WLS(X=SRC, y=OP, sigma=1/OPunc**2)
        station.reg[OPtype] = regr
        station.OPi.loc[:, OPtype] = regr.params[1:]
        station.OPi.loc[:, "SD_"+OPtype] = regr.bse[1:]
        station.OPi.sort_index(inplace=True)
        
        # Bootstrap solution
        pred = pd.DataFrame(index=SRC.index)
        for i in range(NBOOT):
            params = regr.bse * np.random.randn(len(regr.params)) + regr.params
            pred[i] = (params*SRC).sum(axis=1) + params["const"]
        station.OPmodel_unc[OPtype] = pred.std(axis=1)
        station.OPmodel[OPtype] = regr.params["const"]+(station.SRC * station.OPi[OPtype]).sum(axis=1)

    sto[name] = station


# In[33]:


dfPMF_all.station.unique()


# In[37]:


df = pd.DataFrame()
for s in sto.keys():
    df = pd.concat([df, sto[s].OPi])
OPi = pd.DataFrame()
for source in df.index.unique():
    OPi.loc[source, "AAv"] = df.loc[source, "AAv"].mean()
    OPi.loc[source, "DTTv"] = df.loc[source, "DTTv"].mean()
    #OPi.loc[source, "DCFHv"] = df.loc[source, "DCFHv"].mean()
OPi


# In[38]:


# get_ipython().run_line_magic('matplotlib', 'inline')
from scripts.plot_utility import *
sns.set()
sns.set_context("poster")
plot_coeff_all_boxplot(sto, list_OPtype)


# In[39]:


# get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_context("talk")
for OPtype in list_OPtype:
    for station in list_station: 
        if not OPtype in sto[station].OP.columns:
            continue
        if sto[station].OP[OPtype].empty or sto[station].OPmodel[OPtype].isnull().all():
            continue
        #if "Talence" in station:
        #    continue
        print(station)
        plot_station(sto[station], OPtype)
        #plot_save(OPtype+station, "/home/webersa/")
        


# In[40]:


sns.set_context("talk")
for station in sto.values():
    #plot_barplot_contribution(station, ["AAv", "DTTv"])
    plot_barplot_contribution(station, list_OPtype, normalize=False)

for station in sto.values():
    #plot_barplot_contribution(station, ["AAv", "DTTv"])
    plot_barplot_contribution(station, list_OPtype, normalize=True)
   


# In[7]:


for name in ["Chamonix","GRE-fr","MRS-5av"]:
    plot_barplot_contribution(sto[name], list_OPtype=list_OPtype, normalize=False)

