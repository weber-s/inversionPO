import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import pickle
from scipy import linalg
from scipy import polyfit
# from statsmodels.sandbox.regression.predstd import wls_prediction_std

sns.set_context("talk")

def add_season(df, copy=False):
    """
    Add a season column to the DataFrame df from its indexes.
    
    copy: Boolean, default False
        either or not copy the initial dataframe
    """
    
    month_to_season = np.array([
        None,
        'DJF', 'DJF',
        'MAM', 'MAM', 'MAM',
        'JJA', 'JJA', 'JJA',
        'SON', 'SON', 'SON',
        'DJF'
    ])
    
    if copy:
        df_tmp = df.copy()
        df_tmp["season"] = month_to_season[df.index.month]
        return df_tmp
    else:
        df["season"] = month_to_season[df.index.month]
        return

def renameIndex():
    rename={
        "OP_Bio._burning": "Bio. burning",
        "OP_Industrial": "Industrial",
        "OP_Mineral_dust": "Mineral dust",
        "OP_Nitrate_rich": "Nitrate-rich",
        "OP_Primary_bio": "Primary bio",
        "OP_Sea_road_salt": "Sea/road salt",
        "OP_Secondary_bio": "Secondary bio",
        "OP_Sulfate_rich": "Sulfate-rich",
        "OP_Vehicular": "Vehicular",
        "OP_AOS_dust": "AOS/dust",
        "OP_Débris_végétaux": "Débris végétaux",
        "OP_Chlorure": "Chlorure"
        }
    return rename

def sitesColor():
    """
    Colors for the sites. Follows mpl.colors.TABLEAU_COLORS
    """
    color ={
        "Marnaz": "#1f77b4",
        "Passy": "#ff7f0e",
        "Chamonix": "#2ca02c",
        "Frenes": "#d62728",
        "Nice": "#9467bd",
        "PdB": "#8c564b",
        "Marseille": "#e377c2",
        "ANDRA": "#7f7f7f"
    }
    color = pd.DataFrame(index=["color"], data=color)
    return color

def sourcesColor():
    color ={
        "Vehicular": "#000000",
        "Vehicular_ind": "#111111",
        "Vehicular_dir": "#333333",
        "Oil/Vehicular": "#000000",
        "Bio_burning": "#92d050",
        "Bio_burning1": "#92d050",
        "Bio_burning2": "#bbd020",
        "Sulfate_rich": "#ff2a2a",
        "Nitrate_rich": "#ff7f2a",
        "Secondary_bio": "#8c564b",
        "Marine/HFO": "#8c564b",
        "Marine_bio": "#fc564b",
        "HFO": "#70564b",
        "Marine": "#33b0f6",
        "Marin": "#33b0f6",
        "Salt": "#00b0f0",
        "Aged_salt": "#00b0ff",
        "Primary_bio": "#ffc000",
        "Biogenique": "#ffc000",
        "Biogenic": "#ffc000",
        "Dust": "#dac6a2",
        "Industrial": "#7030a0",
        "Indus/veh": "#7030a0",
        "Arcellor": "#7030a0",
        "Siderurgie": "#7030a0",
        "Plant_debris": "#2aff80",
        "Débris végétaux": "#2aff80",
        "Choride": "#80e5ff",
        "PM other": "#cccccc",
        "nan": "#ffffff"
    }
    color = pd.DataFrame(index=["color"], data=color)
    return color

def plot_save(name, OUTPUT_DIR, fmt="png"):
    """
    Save the current figure to OUPUT_DIR with the name 'name'
    Formats are passed in the fmt array. Default is png.
    """
    for k in fmt:
        if k == 'svg':
            plt.savefig(OUTPUT_DIR+"svg/"+name+".svg")
        elif k == 'pdf':
            plt.savefig(OUTPUT_DIR+"pdf/"+name+".pdf")
        else:
            plt.savefig(OUTPUT_DIR+name+"."+k) 


def plot_corr(df,title=None, alreadyDone=False, ax=None, **kwarg):
    """
    Plot the correlation heatmap of df.
    This function use the seaborn heatmap function and simply rotate the labels
    on the axes.
    """
    if ax is None:
        f, ax = plt.subplots()
        kwarg["ax"] = ax
    if "vmax" not in kwarg:
        kwarg["vmax"]=1
    if "square" not in kwarg:
        kwarg["square"]=True
    
    if alreadyDone:
        sns.heatmap(df,**kwarg)
    else:
        sns.heatmap(df.corr(),**kwarg)
    ax.set_yticklabels(df.index[::-1],rotation=0)               
    ax.set_xticklabels(df.columns,rotation=-90)               

    if title is not None:
        ax.set_title(title)        
    elif hasattr(df,"name"):
        ax.set_title(df.name)

    return ax

def plot_scatterReconsObs(ax, station=None, OPtype=None, obs=None, model=None, p=None,
                          pearsonr=None, xerr=None, yerr=None):
    """
    Scatter plot between the observation and the model.
    """
    if station is not None:
        idx = station.OP[OPtype].index.intersection(station.OPmodel[OPtype].index)
        obs     = station.OP.loc[idx, OPtype]
        model   = station.OPmodel.loc[idx, OPtype]
        p       = station.get_ODR_result(OPtype)
        pearsonr= station.pearsonr[OPtype][0]
        xerr    = station.OP.loc[idx, "SD_"+OPtype]
        yerr    = station.OPmodel_unc.loc[idx, OPtype]
    
    # pd.concat((obs,model), axis=1).plot(ax=ax,x=[0], y=[1], kind="scatter")
    ax.errorbar(x=obs, y=model, xerr=xerr, yerr=yerr, label="", 
                elinewidth=1,
                ecolor="k",
                linestyle='None',
                marker="o",
                markersize=6,
                markeredgecolor="white",
                markeredgewidth=1)

    ax.set_aspect("equal","box")
    ax.plot([0,obs.max()],[0, obs.max()], '--', label="y=x")
    ax.plot([0,obs.max()],[p[1], p[0]*obs.max()+p[1]], label="linear fit")
    print(pearsonr)
    ax.text(0.05,0.75,
            "y={:.2f}x{:+.2f}\npearson r={:.2f}".format(p[0],p[1],pearsonr),
            transform=ax.transAxes)
    ax.set_xlabel("Obs.")
    ax.set_ylabel("Recons.")
    ax.set_title("obs. vs reconstruction")
    
    ax.axis("square")
    ax.set_aspect(1./ax.get_data_ratio())
    ticks = ax.get_yticks()
    l=ax.legend(loc="lower right")
    l.draw_frame(False)

def plot_station_sources(station,**kwarg):
    """
    Plot the mass contrib (piechart), the scatter plot obs/recons, the
    intrinsic OP and the contribution of the sources/species (TS + piechart).
    TODO: it's ugly...
    """
    plt.figure(figsize=(17,8))
    # Mass contribution (pie chart)
    ax=plt.subplot(2,3,1)
    plot_contribPie(ax, station.CHEM)
    # Bar plot of coeff for the OP
    ax=plt.subplot(2,3,2)
    plot_coeff(station,ax)
    plt.ylabel("OP [nmol/min/µg]")
    # Scatter plot obs/recons.
    ax=plt.subplot(2,3,3)
    plot_scatterReconsObs(ax, station.OP, station.model, station.p, station.pearson_r)
    # time serie reconstruction/observation
    ax=plt.subplot(2,3,(4,5))
    plot_ts_reconstruction_OP(station,ax=ax)
    plt.legend(mode="expand", bbox_to_anchor=(0.5,-0.1))
    # OP contribution (pie chart)
    ax=plt.subplot(2,3,6)
    plot_contribPie(ax, (station.CHEM*station.mu).sum())

    plt.subplots_adjust(top=0.95, bottom=0.16, left=0.07, right=0.93)

def plot_timeserie_obsvsmodel(station, OPtype, ax=None, **kwarg):
    """Plot the time series obs/model"""
    if ax == None:
        ax = plt.subplot()

    idx = station.OP[OPtype].index.intersection(station.OPmodel[OPtype].index)
    ax.errorbar(idx,
                station.OP.loc[idx, OPtype],
                yerr=station.OP.loc[idx, "SD_"+OPtype], 
                color="#1f77b4",
                ecolor="black",
                elinewidth=1,
                fmt="-o",
                markersize=6,
                linewidth=2,
                label="Obs.",
                zorder=5)
    if station.OPmodel_unc[OPtype] is not None:
        ax.fill_between(idx, 
                        station.OPmodel.loc[idx, OPtype] - station.OPmodel_unc.loc[idx, OPtype],
                        station.OPmodel.loc[idx, OPtype] + station.OPmodel_unc.loc[idx, OPtype],
                        alpha=0.4, edgecolor='#FF7F0E', facecolor='#FF7F0E',
                        zorder=1)
    ax.plot_date(idx,
                 station.OPmodel.loc[idx, OPtype], "r-*",
                 linewidth=2, label="Recons.",zorder=10, color="#FF7F0E")
    ax.axis(ymin=max(ax.axis()[2],-1))
    ax.set_ylabel("{OP} loss\n[nmol/min/m³]".format(OP=OPtype[:-1]))
    plt.title("{station} {OPt}".format(station=station.name, OPt=OPtype))
    handles, labels = ax.get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    l=ax.legend(handles, labels) # in order to have Obs 1st
    l.draw_frame(False)


def plot_station(station,OPtype,**kwarg):
    """
    Plot the time series obs & recons, the scatter plot obs/recons, the
    intrinsic OP and the contribution of the sources/species.
    """
    plt.figure(figsize=(17,8))
    # time serie reconstruction/observation
    ax=plt.subplot(2,3,(1,3))
    plot_timeserie_obsvsmodel(station, OPtype, ax=ax)
    # scatter plot reconstruction/observation
    ax=plt.subplot(2,3,4)
    plot_scatterReconsObs(ax, station, OPtype=OPtype)
    # factors contribution
    ax=plt.subplot(2,3,5)
    plot_coeff(station, OPtype=OPtype, ax=ax)
    plt.ylabel("OP [nmol/min/µg]")
    # Pie chart
    ax=plt.subplot(2,3,6)
    plot_contribPie(ax, station, OPtype=OPtype, **kwarg)

    plt.subplots_adjust(top=0.95, bottom=0.16, left=0.07, right=0.93)

def plot_contribPie(ax, station, OPtype=None, fromSource=True, title=None, **kwarg):
    """
    Plot contributions of the sources to the OP in a Pie chart
    The contributions is G*m.
    """
    # check if station is an object or a DataFrame
    if isinstance(station, pd.DataFrame):
        df = station.sum()
    else:
        if not station.OP[OPtype].any():
            return
        df = pd.Series((station.SRC * station.OPi[OPtype]).sum(), name=station.name)
    
    # add color to the sources
    c = sourcesColor()
    cols = c.loc["color",df.index].values
    kwarg["colors"] = cols

    ax.set_aspect('equal')
    # plot the pie plot
    p = df.plot.pie(ax=ax,
                    shadow=False,
                    startangle=90,
                    **kwarg)
    
    if title is not None:
        ax.set_title(title)
    ax.set_ylabel("")

def plot_coeff(station, OPtype, yerr=None, ax=None):
    """Plot a bar plot of the intrinsic OP of the sources for the station(s)"""
    if ax==None:
        ax = plt.subplots(row=len(OPtype_list), columns=len(station.keys()))

    c = sitesColor()
    cols = list()
    if hasattr(station, "name"):
        cols.append(c.loc["color"][station.name])
        station.OPi[OPtype].plot.bar(ax=ax, yerr=station.OPi["SD_"+OPtype],
                                     legend=False, color=cols)
    # try:
    #     for s in stations:
    #         cols.append(c.loc["color"][s])
    #     stations.plot.bar(ax=ax, yerr=yerr, legend=False, color=cols,rot=30)
    # except TypeError:
    #     cols.append(c.loc["color"][stations.name])
    #     stations.OPi.plot.bar(ax=ax, yerr=stations.covm, legend=False, color=cols)

def plot_ts_contribution_OP(station,OPtype=None,saveDir=None, ax=None):
    """
    Plot the time serie contribution of each source to the OP.
    station can be the name of the station or a Station object.
    If station is a string, then the saveDir variable must be the path to the
    directory where the file is saved.
    The file name must be in the format 
        {station name}_contribution_{OPtype}.csv
    """
    
    if isinstance(station, str):
        if saveDir == None:
            print("ERROR: the 'saveDir' argument must be completed")
            return
        print("Use the saved results")
        title = station 
        fileName = saveDir+station+"_contribution_"+OPtype+".csv"
        df = pd.read_csv(fileName,index_col="date", parse_dates=["date"])
    else:
        df = station.CHEM * station.OPi
        title = station.name

    if ax == None:
        ax = plt.subplot()

    c = sourcesColor()
    cols = c.loc["color",df.columns].values
    
    df.plot(title=title, color=cols, ax=ax)
    ax.set_ylabel(OPtype)
    return

def plot_ts_reconstruction_OP(station, OPtype=None, OPobs=None, saveDir=None, ax=None):
    """
    Plot a stacked barplot of for the sources contributions to the OP
    """
    if ax == None:
        f, ax = plt.subplots(1, figsize=(10,5))

    if isinstance(station, str):
        if saveDir == None or OPobs == None:
            print("ERROR: the 'saveDir' and 'OPobs' arguments must be completed")
            return
        title = station 
        fileName = saveDir+station+"_contribution_"+OPtype+".csv"
        OP = OPobs
        df = pd.read_csv(fileName,index_col="date", parse_dates=["date"])
    else:
        df = station.CHEM * station.OPi
        OP = station.OP.values
        OPunc = station.OPunc.values
        title = station.name
    
    c = sourcesColor()
    cols = c.loc["color",df.columns].values
    
    # Date index
    x = df.index
    x.to_datetime()

    # Width
    # Set it to 1.5 when no overlapping, 1 otherwise.
    width = np.ones(len(x))*1.5
    deltal = x[1:]-x[:-1]
    deltal = deltal.append(pd.TimedeltaIndex([10,],'D'))
    deltar = pd.TimedeltaIndex([3],'D')
    deltar = deltar.append(x[1:]-x[:-1])
    width[deltal < np.timedelta64(2,'D')] = 1
    width[deltar < np.timedelta64(2,'D')] = 1
    
    # Stacked bar plot
    count = 0
    for i in range(df.shape[1]):
        bottom=df.ix[:,0:count].sum(axis=1)
        count += 1
        ax.bar(x, df[df.columns[i]],
               bottom=bottom,
               label=df.columns[i],
               width=width,
               color=c[df.columns[i]])
    # OP observation
    ax.errorbar(x, OP, OPunc, fmt='ob', ecolor="black", elinewidth=1, markersize=3, label="OP obs.")

    # legend stuff
    ncol = int((len(df.columns)+1)/2)
    nrow = (len(df.columns)+1)/ncol
    if nrow > 2:
        ncol += 1
    ax.legend(loc="center",ncol=ncol,bbox_to_anchor=(0.5,-0.2))
    ax.set_title(title)
    ax.set_ylabel(OPtype)

    plt.subplots_adjust(top=0.90, bottom=0.20, left=0.10, right=0.90)
    return

def plot_seasonal_contribution(station, OPtype=None,
                               saveDir=None,CHEMorOP="OP",**kwarg):
    """
    Plot a stacked bar plot of the normalized contribution of the source to the
    OP.
    """

    # first, check of station is a string
    # if so, then load the associated Station class (previously saved).
    # if isinstance(station, str):
    #     with open(saveDir+"/"+station+"_"+OPtype+".pickle","rb") as f:
    #         station = pickle.load(f)
    if CHEMorOP=="CHEM":
        df = station.SRC.copy()
    else:
        df = station.OPi[OPtype] * station.SRC

    add_season(df)

    df_grouped = df.groupby("season").sum()
    ordered_season = ["DJF","MAM","JJA","SON"]
    df_grouped = df_grouped.reindex(ordered_season)

    # selection the colors we have in the sources
    colors  = sourcesColor()
    c       = colors.loc["color", df_grouped.columns]
    # plot the stacked normalized bar plot
    df   = (df_grouped.T / df_grouped.sum(axis=1))
    df.index = [l.replace("_"," ") for l in df.index]
    axes = df.T.plot.bar(stacked=True,
                         rot=0,
                         color=c,
                         **kwarg)
    if "ax" in kwarg:
        ax = kwarg["ax"]
    else:
        ax = plt.gca()
    # ax.legend(loc="center",ncol=round(len(df_grouped.columns)/2)+1, bbox_to_anchor=(0.5,-0.3))
    ax.legend(loc="center",ncol=4, bbox_to_anchor=(0.5,-0.3))

    # ax.set_ylabel("Ncontribution (normalized)")
    # ax.set_ylabel("Mass contribution (normalized)")
    # plt.title(station.name+" (DTTv)")
    ax.set_title(station.name+" "+OPtype)
    plt.subplots_adjust(top=0.90, bottom=0.10, left=0.15, right=0.85)

def plot_anual_contribution(station, **kwarg):
    
    df = pd.DataFrame(columns=["PM mass", "OP DTTv", "OP AAv"])
    # mass 
    df["PM mass"] = station.SRC.sum()
    # OP DTT and OP AAv
    df["OP DTTv"] = (station.OPi["DTTv"] * station.SRC).sum()
    df["OP AAv"] = (station.OPi["AAv"] * station.SRC).sum()

    # selection the colors we have in the sources
    colors  = sourcesColor()
    c       = colors.loc["color", df.T.columns]
    # plot the stacked normalized bar plot
    df   = df / df.sum()
    df.index = [l.replace("_"," ") for l in df.index]
    axes = df.T.plot.bar(stacked=True,
                         rot=0,
                         color=c,
                         **kwarg)
    axes.set_title("Anual")



def plot_seasonal_contribution_boxplot(station, OPtype=None, saveDir=None,**kwarg):
    """
    Plot a boxplot contribution of the source to the OP per season.
    """

    # first, check of station is a string
    # if so, then load the associated Station class (previously saved).
    if isinstance(station, str):
        with open(saveDir+"/"+station+"_"+OPtype+".pickle","rb") as f:
            station = pickle.load(f)

    df = station.OPi * station.CHEM

    add_season(df)
    season = np.array(['DJF', 'MAM', 'JJA','SON'])
    df["ordered"] = season[df["season"]]
    ordered_season = ["DJF","MAM","JJA","SON"]
    
    # selection the colors we have in the sources
    colors  = sourcesColor()
    c       = colors.loc["color", df.columns]
    # plot the boxplot
    df_long = pd.melt(df,"season",var_name="source", value_name="OP")
    ax = sns.boxplot("season", y="OP",hue="source",data=df_long,palette=c)

    if "title" in kwarg:
        plt.title(kwarg["title"])

def plot_regplot(X, Y, title=None):
    """
    Plot the regression plot of the OP vs the source/chem using seaborn.regplot
    """

    df = pd.concat([X, Y],axis=1)
    g = sns.PairGrid(df, y_vars=Y.columns,
                     x_vars=df.drop(Y.columns, axis=1).columns)                               
    g.map(sns.regplot, scatter_kws={'s':4}, line_kws={'linewidth':2})

    # set label nicely
    for axes in g.axes:
        [item.set_fontsize(10) for item in axes[0].get_yticklabels()]
        [ax.set_ylim(bottom=-1) for ax in axes]
    for a in g.axes[-1]:
        a.set_xlabel(a.get_xlabel().replace("_"," "), fontsize=10)
        [item.set_fontsize(10) for item in a.get_xticklabels()]
    if title is not None:
        plt.suptitle(title)

def plot_all_coeff(list_station, OP_type, SAVE_DIR, ax=None):
    """
    plot a boxplot + a swarmplot plot of the coefficients
    stations: list name of the station to plot.
    """
    if ax is None:
        ax = plt.gca()
    
    # get all the station and sources
    stations_dict = dict()
    coeff = pd.DataFrame()
    for name in list_station:
        with open(SAVE_DIR+"/"+name+OP_type+".pickle", "rb") as h:
            stations_dict[name] = pickle.load(h)
    coeff = coeff.join([stations_dict[name].OPi for name in list_station], how="outer")
    source = list(coeff.index)

    # coeff.drop(["Marine/HFO"], inplace=True)
    # make it in a categorical dataframe
    multiIndex = pd.MultiIndex.from_product([source,list_station],names=["source","site"])
    df = pd.DataFrame(data=coeff.values.reshape(coeff.values.size,),
                      index=multiIndex,
                      columns=["val",])

    # get color palette
    colors = sitesColor()
    colors = colors[list_station]
    palette = colors.values.reshape(len(list_station),)

    # ... then plot it
    sns.boxplot(data=coeff.T, color="white",
                ax=ax)

    sns.swarmplot(x="source", y="val", hue="site", 
                  data=df.reset_index(),
                  palette=palette, 
                  size=8, edgecolor="black",
                  ax=ax)
    ax.legend("")

    ax.set_title(OP_type)
    ax.set_xlabel("")
    ax.set_ylabel("nmol/min/µg")

def plot_monthly_OP_boxplot(OP):
    """
    Plot the time series of the OP, one boxplot per month.
    OP: a DataFrame with "AAv" and "DTTv" column
    """

    f = plt.figure(figsize=[11.25,2.8])
    
    df = pd.DataFrame(data={"val":OP["DTTv"].values, "type":"DTTv",
                            "date":OP.index})    
    df=df.append(pd.DataFrame(data={"val":OP["AAv"].values,"type":"AAv",
                                    "date":OP.index}))
    df["type"]=df["type"].astype("category")  
    df.index = df["date"]
    ax=sns.boxplot(x=df.index.month, y=df["val"], hue=df["type"])

    
    ax.set_xticklabels(["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"])
    ax.set_ylabel("OP [nmol/min/m³]")
    ax.legend(loc="center", bbox_to_anchor=(1.06,0.5))

    OP.plot(marker="*", figsize=[11.25,2.8], rot=0)

def plot_seasonal_OP_source(list_station, list_OPtype, station_dict=None,
                            saveDir=None):
    """
    Plot a barplot of the seasonal OP_src contributions

    parameters
    ----------

    list_station: array-like, the list of the station to plot

    list_OPtype: array-like, the list of the OP type (AA, DTT,...) to plot

    station_dict: default None, a dict with `Station` object.

    saveDir: default None, if staction_dict is None, use to load the Station.
    """

    if staction_dict is None:
        station_dict = dict()
        for OPtype in list_OPtype:
            station_dict[OPtype] = dict()
            for name in list_station:
                with open(saveDir+"/"+name+"_"+OPtype+".pickle","rb") as f:
                    station = pickle.load(f)
                station_dict[OPtype][name] = station

    for OPtype in list_OPtype:
        coeff = coeff.join([stations_dict[name].OPi for name in list_station], how="outer")
        source = list(coeff.index)

def plot_scatterReconsObs_all(sto,list_OPtype,list_station):
    """
    Plot the scatter plot obs/reconstructed for all the stations.
    """
    f, axes = plt.subplots(nrows=len(list_OPtype), ncols=len(list_station),
                           figsize=(14,4))
    for i, OPtype in enumerate(list_OPtype):
        for j, name in enumerate(list_station):
            plot_scatterReconsObs(axes[i][j], sto[OPtype][name].OP,
                                  sto[OPtype][name].model, sto[OPtype][name].p,
                                  sto[OPtype][name].pearson_r)

            if i==0:
                axes[i][j].set_title(name)
                axes[i][j].set_xlabel("")
            else:
                axes[i][j].set_title("")
            
            if j==0:
                axes[i][j].set_ylabel(OPtype+"\nReconstructed")
            else:
                axes[i][j].set_ylabel("")

    plt.subplots_adjust(top=0.95,bottom=0.11,left=0.07,right=0.99)

def plot_compare_MassOP(stations, list_OPtype, source=None):
    """
    Compare the contribution of the source `source` 
    """
    
    df = pd.DataFrame()
    for station in stations["DTTv"].values():
        df.loc["Mass",station.name] = (station.CHEM.sum()\
                               /station.CHEM.sum().sum())[source]    
    for OPtype in list_OPtype:
        for station in stations[OPtype].values():
            df.loc[OPtype,station.name] = ((station.CHEM*station.OPi).sum()\
                         /((station.CHEM*station.OPi).sum().sum()))[source]     
    return df

def plot_normalized_contribution(station, list_OPtype):
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
    axes[-1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    axes[-1].set_xlabel(" ")

def plot_residualvsobs(station, OPtype=None, **kwarg):

    idx = station.OPmodel.index.intersection(station.OP.index)
    g = sns.JointGrid(x=station.OP.loc[idx, OPtype],
                      y=station.reg[OPtype].resid.loc[idx])
    g.plot_joint(plt.errorbar, 
                 xerr=station.OP.loc[idx, "SD_"+OPtype],
                 yerr=station.OPmodel_unc.loc[idx, OPtype],
                 elinewidth=1,
                 ecolor="k",
                 linestyle='None',
                 marker="o",
                 markersize=6,
                 markeredgecolor="white",
                 markeredgewidth=1,
                 **kwarg)
    g.plot_marginals(sns.distplot, kde=False, **kwarg)
    g.ax_marg_x.remove()
    g.ax_joint.set_ylabel("Obs. - Model [nmol/min/m$^3$]")
    g.ax_joint.plot([0,g.ax_joint.axis()[1]], [0,0], "--k")
    g.ax_joint.set_title(OPtype)
    g.ax_joint.set_xlabel("Obs. [nmol/min/m$^3$]")
    return g

def print_correlation_obs_predicted(sto, list_station, list_OPtype):
    """
    Output a latex table for the correlation coefficient and regresion line
    between OP and predicted.
    """
    print("\\toprule\nOP & Station & $ax+b$ & $r^2$ & \\OP{{}} range \\\\")
    for OPtype in list_OPtype:
        print("\\midrule\n\\multirow{{{}}}{{*}}{{\\{}}}".format(len(list_station),OPtype))
        for name in list_station:
            s = sto[OPtype][name]
            print(" & {:8} & ${:4.2f}x{:+3.2f}$ & {:3.2f} & {:3.2f} to {:3.2f}\\\\".format(s.name, s.p[0],s.p[1], s.pearson_r[0,1],
                                       s.OP.min(), s.OP.max()))
    print("\\bottomrule")

def print_coeff(sto, list_station, list_OPtype):
    """
    Build a dataframe with the `coeff±bse (pvalue)` string for each site and
    source
    """

    source = list()
    for OPtype in list_OPtype:
        for name in list_station:
            source = source + [a for a in sto[OPtype][name].CHEM.columns if a not in source]
    source.sort()

    # Multiindex part
    index1 = list_OPtype * len(source)
    index1.sort()
    arrays = [index1,source*len(list_OPtype)]
    tuples = list(zip(*arrays))
    index = pd.MultiIndex.from_tuples(tuples, names=['OP type', 'Sources'])
    
    coeff = pd.DataFrame(index=index, columns=list_station,data=0.0)
    p = pd.DataFrame(index=index, columns=list_station,data=0.0)
    bse = pd.DataFrame(index=index, columns=list_station,data=0.0)

    for name in list_station:
        for OPtype in list_OPtype:
            s = sto[OPtype][name]
            coeff.loc[OPtype][name]=s.reg.params
            p.loc[OPtype][name]=s.reg.pvalues
            bse.loc[OPtype][name]=s.reg.bse
    
    coeff.fillna(0,inplace=True) 
    p.fillna(0,inplace=True) 
    bse.fillna(0,inplace=True) 

    df = coeff.applymap(lambda x: "{:.2f}±".format(x))\
            +bse.applymap(lambda x: "{:.2f}".format(x))\
            +p.applymap(lambda x: " ({:.2f})".format(x))
    
    # replace 0 by --- if the source is not in the site
    for OPtype in list_OPtype:
        for name in list_station:
            for s in source:
                if s not in sto[OPtype][name].CHEM.columns:
                    df.loc[(OPtype,s),name]="---"
                elif s not in sto[OPtype][name].reg.params.index:
                    df.loc[(OPtype,s),name]="0"

    # replace _ by a space
    df.index.set_levels([x.replace("_"," ") for x in df.index.get_level_values(level=1)],
                        level=1, inplace=True)
    
    print(df.to_latex(multirow=True, column_format="ll"+"c"*len(list_station)))

    return df

def print_equation(Station):
    """
    Print the equation relathionship between the OP and the sources
    """
    coeff = ""
    for src in Station.OPi.index:
        if not np.isnan(Station.OPi[src]):
            coeff = coeff + " + " + str(round(Station.OPi[src],2)) + "×" + src
    print("{OPtype} = {const}{coeff}".format(OPtype=Station.OPtype,
                                                const=round(Station.reg.params[0],2),
                                                coeff=coeff))
