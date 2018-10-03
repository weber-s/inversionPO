import pandas as pd
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime as dt

def get_df():
    def hasDate(c):
        return pd.notnull(pd.to_datetime(c, errors="coerce"))

    filePM="~/Documents/BdD/BdD_PM/BdD_PM_all.xlsx"
    filePO="~/Documents/BdD/BdD_PM/BdD_PO_all.xlsx"

    naliste=["NA","n.a.","n.a","-","na","nd"]

    wbChem = pd.read_excel(filePM, sheet_name=None, header=0, na_values=naliste)
    wbPO = pd.read_excel(filePO, sheet_name=None, header=1, na_values=naliste)

    dfChem = pd.concat(wbChem)
    dfChem.index.names=["station","number ID"]
    dfChem.dropna(axis=1, how="all", inplace=True)

    dfPO = pd.concat(wbPO)
    dfPO.rename_axis({"date.prelevement":"date", "echantillon":"sample ID"}, axis=1, inplace=True)
    dfPO.index.names=["station","number ID"]
    dfPO.dropna(axis=1, how="all", inplace=True)

    isDateChem = hasDate(dfChem.date)
    isDatePO = hasDate(dfPO.date)


    keep_PO_col = ["station", "commentary", "sample ID", "ville", "date", "date.analyse",
                   "analysis", "PM_µg/m3", "PO_DTT_µg", "SD_PO_DTT_µg", "PO_DTT_m3",
                   "SD_PO_DTT_m3", "PO_AA_µg", "SD_PO_AA_µg", "PO_AA_m3",
                   "SD_PO_AA_m3", "nmol_H2O2.µg-1", "SD_PO_DCFH_µg",
                   "nmol_H2O2equiv.m-3 air", "SD_PO_DCFH_m3"]

    stationChem = set(dfChem.index.get_level_values("station"))
    stationPO = set(dfPO.index.get_level_values("station")) - {
        "Filtres de references",
        "Récapitulatif des analyses",
        "Wood burning Suisse"
    }
    stationAll = set(stationChem)
    stationAll.update(stationPO)
    stationOnlyPO = stationAll-set(stationChem)
    stationOnlyChem = stationAll-set(stationPO)
    stationCommon = stationAll-stationOnlyChem -stationOnlyPO                                     

    # FIXME: it should be a better idea than this big loop...
    df = pd.DataFrame()
    for s in stationAll:
        print(s)
        dfTmpChem = pd.DataFrame()
        dfTmpPO = pd.DataFrame()
        if s in stationChem:
            dfTmpChem = dfChem.loc[s]
            dfTmpChem["station"] = s
            dfTmpChem = dfTmpChem[hasDate(dfTmpChem.date)]
            dfTmpChem["date"] = pd.DatetimeIndex(dfTmpChem["date"]).normalize()
            dfTmpChem.set_index(["station", "date", "analysis"], inplace=True)
        if s in stationPO:
            dfTmpPO = dfPO.loc[s]
            dfTmpPO["station"] = s
            dfTmpPO = dfTmpPO[hasDate(dfTmpPO.date)]
            dfTmpPO = dfTmpPO[keep_PO_col]
            isHeader = dfTmpPO["commentary"]=="header"
            dfTmpPO = dfTmpPO[~isHeader]
            dfTmpPO["date"] = pd.DatetimeIndex(dfTmpPO["date"]).normalize()
            dfTmpPO.set_index(["station", "date", "analysis"], inplace=True)
        if dfTmpChem.shape == (0,0):
            dfTmp = dfTmpPO
        elif dfTmpPO.shape == (0,0):
            dfTmp = dfTmpChem
        else:
            dfTmp = dfTmpChem.join(dfTmpPO, how="outer", lsuffix="_chem",
                                   rsuffix="_PO")
        df = df.append(dfTmp)


    isDate = hasDate(df.index.get_level_values("date"))
    df.dropna(axis=1, how="all", inplace=True)
    df_val = df[isDate]
    # there are some value in date.analyse that are "Fait par aude", so we can't
    # convert this column to date...
    dateColumn=["date", "date IC-sucres", "date metals"]
    textColumn=["sample ID_chem","commentary","sample ID_PO", "commentary_PO","commentary_chem","analysis","programme"]
    boolColumn=["big serie"]
    realColumn=list(set(df_val.columns) - set(dateColumn+textColumn+boolColumn))

    # ensure date is a date
    for d in dateColumn:
        if d in df_val.columns and df_val[d].dtypes != 'datetime64[ns]':
            df_val.loc[:,d] = pd.to_datetime(df_val[d])
    # ensure bool is a bool
    df_val.loc[:,boolColumn]=df_val.loc[:,boolColumn].fillna(False).astype(bool)
    # replace string by "error code" and ensure float is float
    df_val.replace({"<QL":-1,"<DL":-2}, inplace=True)
    df_val.loc[:,realColumn]=df_val.loc[:,realColumn].apply(pd.to_numeric, errors='coerce')
    return df_val




#dfc = get_df()
dfc = pd.read_pickle("~/Documents/BdD/BdD_PM/df_PO_PM.pickle")

stations=sorted(dfc.index.get_level_values("station").unique())
df=pd.DataFrame(columns=["name","OP","chem","datemin", "datemax"])
for s in stations:
    d=dfc.xs(s, level="station")
    if "PM10" in d.index.get_level_values("analysis"):
        df.loc[s,"name"] = s
        df.loc[s,"OP"]   = pd.np.sum(pd.notnull(d["PO_DTT_m3"]))
        df.loc[s,"chem"] = pd.np.sum(pd.notnull(d["OC"]))
        df.loc[s,"datemin"]= d.index.get_level_values("date").min()
        df.loc[s,"datemax"]= d.index.get_level_values("date").max()


ax = plt.subplot()
for i, s in enumerate(df.sort_values("datemin")["name"]):
    start = mdates.date2num(df.loc[s,"datemin"].to_pydatetime()) 
    end = mdates.date2num(df.loc[s,"datemax"].to_pydatetime()) 
    width = end - start
    mean = (end+start)/2
    
    rect = Rectangle((start, i+0.1), width, 0.8, color='blue')
    
    ax.add_patch(rect)
    plt.text(mean, i+0.4, s, horizontalalignment="center")

ax.set_xlim((mdates.date2num(df.loc[:,"datemin"].min().to_pydatetime())-30,
            mdates.date2num(df.loc[:,"datemax"].max().to_pydatetime())+30))
ax.set_ylim(0,i+1)

# assign date locator / formatter to the x-axis to get proper labels
locator = mdates.AutoDateLocator(minticks=3)
formatter = mdates.AutoDateFormatter(locator)
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(formatter)
