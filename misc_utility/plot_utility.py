import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import pickle
from scipy import linalg
from scipy import polyfit
from statsmodels.sandbox.regression.predstd import wls_prediction_std

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

def plot_scatterReconsObs(ax, obs, model, p, r2):
    """
    Scatter plot between the observation and the model.
    """
    # pd.concat((obs,model), axis=1).plot(ax=ax,x=[0], y=[1], kind="scatter")
    ax.scatter(x=obs, y=model, label="", edgecolor="white")

    ax.set_aspect("equal","box")
    ax.plot([0,obs.max()],[0, obs.max()], '--', label="y=x")
    ax.plot([0,obs.max()],[p[1], p[0]*obs.max()+p[1]], label="linear fit")
    ax.text(0.05,0.75,"y={:.2f}x{:+.2f}\nr²={:.2f}".format(p[0],p[1],r2[0,1]),transform=ax.transAxes)
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
    plot_contribPie(ax, station.pieCHEM)
    # Bar plot of coeff for the OP
    ax=plt.subplot(2,3,2)
    plot_coeff(station,ax)
    plt.ylabel("OP [nmol/min/µg]")
    # Scatter plot obs/recons.
    ax=plt.subplot(2,3,3)
    plot_scatterReconsObs(ax, station.OP, station.model, station.p, station.r2)
    # time serie reconstruction/observation
    ax=plt.subplot(2,3,(4,5))
    plot_ts_reconstruction_OP(station,ax=ax)
    plt.legend(mode="expand", bbox_to_anchor=(0.5,-0.1))
    # OP contribution (pie chart)
    ax=plt.subplot(2,3,6)
    plot_contribPie(ax, station)

    plt.subplots_adjust(top=0.95, bottom=0.16, left=0.07, right=0.93)

def plot_station(station,OPtype,**kwarg):
    """
    Plot the time series obs & recons, the scatter plot obs/recons, the
    intrinsic OP and the contribution of the sources/species.
    """
    plt.figure(figsize=(17,8))
    # time serie reconstruction/observation
    ax=plt.subplot(2,3,(1,3))
    ax.errorbar(station.OP.index.to_pydatetime(), station.OP,
                yerr=station.OPunc, 
                ecolor="black",
                elinewidth=1,
                fmt="-o",
                markersize=6,
                linewidth=2,
                label="Obs.",
                zorder=5)
    if station.yerr is not None:
        ax.fill_between(station.model.index.to_datetime(), 
                        station.model-station.yerr, station.model + station.yerr,
                        alpha=0.4, edgecolor='#CC4F1B', facecolor='#FF9848',
                        zorder=1)
    ax.plot_date(station.model.index.to_pydatetime(), station.model, "r-*",
                 linewidth=2, label="Recons.",zorder=10, color="#ff5048")
    ax.axis(ymin=max(ax.axis()[2],-1))
    ax.set_ylabel("{OP} loss\n[nmol/min/m³]".format(OP=OPtype[:-1]))
    plt.title("{station} {OPt}".format(station=station.name, OPt=OPtype))
    handles, labels = ax.get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    l=ax.legend(handles, labels) # in order to have Obs 1st
    l.draw_frame(False)
    # scatter plot reconstruction/observation
    ax=plt.subplot(2,3,4)
    plot_scatterReconsObs(ax, station.OP, station.model, station.p, station.r2)
    # factors contribution
    ax=plt.subplot(2,3,5)
    plot_coeff(station, ax=ax)
    plt.ylabel("OP [nmol/min/µg]")
    # Pie chart
    ax=plt.subplot(2,3,6)
    plot_contribPie(ax, station,**kwarg)

    plt.subplots_adjust(top=0.95, bottom=0.16, left=0.07, right=0.93)

def plot_contribPie(ax, station, fromSource=True, title=None, ylabel=None,
                    **kwarg):
    """
    Plot contributions of the sources to the OP in a Pie chart
    The contributions is G*m.
    """
    # check if station is an object or a DataFrame
    if isinstance(station, pd.Series):
        df = station
    else:
        if not(station.hasOP):
            ax.set_aspect('equal')
            p = station.pie.plot.pie(ax=ax, **kwarg)
            ax.set_ylabel("")
            return
        df = station.pie
    


    if fromSource:
        c = sourcesColor()
        cols = c.ix["color",df.index].values
        ax.set_aspect('equal')
        p = df.plot.pie(ax=ax,
                        shadow=False,
                        startangle=90,
                        colors=cols,
                        **kwarg)

    else:
        ax.set_aspect('equal')
        p = df.plot.pie(ax=ax,
                        shadow=False,
                        startangle=90,
                        **kwarg)
    
    ax.set_xlabel(ax.get_xlabel().replace("_"," "))

    labels = df.index

    #for p1, l1 in zip(p[0], labels):
    #    r = p1.r
    #    dr = r*0.1
    #    t1, t2 = p1.theta1, p1.theta2
    #    theta = (t1+t2)/2.
    #    
    #    xc, yc = r/2.*cos(theta/180.*pi), r/2.*sin(theta/180.*pi)
    #    x1, y1 = (r+dr)*cos(theta/180.*pi), (r+dr)*sin(theta/180.*pi)
    #    if x1 > 0 :
    #        x1 = r+2*dr
    #        ha, va = "left", "center"
    #        tt = -180
    #        cstyle="angle,angleA=0,angleB=%f"%(theta,)
    #    else:
    #        x1 = -(r+2*dr)
    #        ha, va = "right", "center"
    #        tt = 0
    #        cstyle="angle,angleA=0,angleB=%f"%(theta,)
    #
    #    annotate(l1,
    #             (xc, yc), xycoords="data",
    #             xytext=(x1, y1), textcoords="data", ha=ha, va=va,
    #             arrowprops=dict(arrowstyle="-",
    #                             connectionstyle=cstyle,
    #                             patchB=p1))

    if title is not None:
        ax.set_title(title)
    ax.set_ylabel("")

def plot_coeff(stations, yerr=None, ax=None):
    """Plot a bar plot of the intrinsic OP of the sources for the station(s)"""
    if ax==None:
        ax = plt.subplots(row=len(OPtype_list), columns=len(stations.keys()))

    c = sitesColor()
    cols = list()
    try:
        for s in stations:
            cols.append(c.ix["color"][s])
        stations.plot.bar(ax=ax, yerr=yerr, legend=False, color=cols,rot=30)
    except TypeError:
        cols.append(c.ix["color"][stations.name])
        stations.m.plot.bar(ax=ax, yerr=stations.covm, legend=False, color=cols)

def plot_ts_contribution_OP(station,OPtype=None,saveDir=None):
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
        df = station.CHEM * station.m
        title = station.name

    c = sourcesColor()
    cols = c.ix["color",df.columns].values
    
    df.plot(title=title, color=cols)
    plt.ylabel(OPtype)
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
        df = station.CHEM * station.m
        OP = station.OP.values
        OPunc = station.OPunc.values
        title = station.name

    c = sourcesColor()
    cols = c.ix["color",df.columns].values
    
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
    plt.legend(loc="center",ncol=ncol,bbox_to_anchor=(0.5,-0.16))
    plt.title(title)
    plt.ylabel(OPtype)

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
    if isinstance(station, str):
        with open(saveDir+"/"+station+"_"+OPtype+".pickle","rb") as f:
            station = pickle.load(f)
    
    if CHEMorOP=="CHEM":
        df = station.CHEM.copy()
    else:
        df = station.m * station.CHEM

    add_season(df)

    df_grouped = df.groupby("season").sum()
    ordered_season = ["DJF","MAM","JJA","SON"]
    df_grouped = df_grouped.reindex(ordered_season)

    # selection the colors we have in the sources
    colors  = sourcesColor()
    c       = colors.ix["color", df_grouped.columns]
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
    ax.legend(loc="center",ncol=round(len(df_grouped.columns)/2)+1, bbox_to_anchor=(0.5,-0.3))

    # ax.set_ylabel("Ncontribution (normalized)")
    # ax.set_ylabel("Mass contribution (normalized)")
    # plt.title(station.name+" (DTTv)")
    ax.set_title(station.name+" "+OPtype)
    plt.subplots_adjust(top=0.90, bottom=0.10, left=0.15, right=0.85)

def plot_seasonal_contribution_boxplot(station, OPtype=None, saveDir=None,**kwarg):
    """
    Plot a boxplot contribution of the source to the OP per season.
    """

    # first, check of station is a string
    # if so, then load the associated Station class (previously saved).
    if isinstance(station, str):
        with open(saveDir+"/"+station+"_"+OPtype+".pickle","rb") as f:
            station = pickle.load(f)

    df = station.m * station.CHEM

    add_season(df)
    season = np.array(['DJF', 'MAM', 'JJA','SON'])
    df["ordered"] = season[df["season"]]
    ordered_season = ["DJF","MAM","JJA","SON"]
    
    # selection the colors we have in the sources
    colors  = sourcesColor()
    c       = colors.ix["color", df.columns]
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

    stations_dict = dict()
    source = list()
    for name in list_station:
        with open(SAVE_DIR+"/"+name+OP_type+".pickle", "rb") as h:
            stations_dict[name] = pickle.load(h)
        source = source + [a for a in stations_dict[name].CHEM.columns if a not in source]
    source.sort()
    
    coeff = pd.DataFrame(index=source)
    for name in list_station:
        coeff[name] = stations_dict[name].reg.params
        # correct the NaN to 0 if the source is present
        for s in source:
            if s in stations_dict[name].CHEM.columns and pd.isnull(coeff[name][s]):
                coeff[name][s] = 0

    # make it in a categorical dataframe
    tmp = pd.DataFrame()
    df = coeff.copy()
    source_cat = list(df.index)*len(df.columns)
    source_cat.sort()
    station_cat = list()
    for j in range(df.values.size):
        station_cat.append(list_station[j%len(df.columns)])

    tmp["val"] = df.values.reshape(df.values.size,)
    tmp["source"] = source_cat
    tmp["station"] = station_cat

    # get color palette
    colors = sitesColor()
    colors = colors[list_station]
    palette = colors.values.reshape(len(list_station),)

    # ... then plot it
    # sns.boxplot(coeff.T, color="white",
    #             ax=ax)

    sns.violinplot(coeff.T, color="white",
                   ax=ax,
                   cut=0)
    sns.swarmplot(x="source", y="val", hue="station", 
                  data=tmp,
                  palette=palette, 
                  size=8, edgecolor="black",
                  ax=ax)
    ax.legend("")


    ax.set_title(OP_type)
    ax.set_xlabel("")
    ax.set_ylabel("nmol/min/µg")

def plot_seasonal_OP_boxplot(OP):
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

def plot_scatterReconsObs_all(sto,list_OPtype,list_station):
    f, axes = plt.subplots(nrows=len(list_OPtype), ncols=len(list_station),
                           figsize=(14,4))
    for i, OPtype in enumerate(list_OPtype):
        for j, name in enumerate(list_station):
            plot_scatterReconsObs(axes[i][j], sto[OPtype][name].OP,
                                  sto[OPtype][name].model, sto[OPtype][name].p,
                                  sto[OPtype][name].r2)

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
            print(" & {:8} & ${:4.2f}x{:+3.2f}$ & {:3.2f} & {:3.2f} to {:3.2f}\\\\".format(s.name, s.p[0],s.p[1], s.r2[0,1],
                                       s.OP.min(), s.OP.max()))
    print("\\bottomrule")

def print_coeff(sto, list_station, list_OPtype):
    """
    Output a latex table for the coefficient 
    """
    
    # get list of source
    import re

    source = list()
    for OP_type in list_OPtype:
        for name in list_station:
            source = source + [a for a in sto[OP_type][name].CHEM.columns if a not in source]
    source.sort()
    
    sourceHeader = " & ".join(source)
    print("\\toprule\nSource &" + " & ".join(list_station) +"\\\\")

    for OP_type in list_OPtype:
        coeff = pd.DataFrame(index=source)
        p     = pd.DataFrame(index=source)
        bse   = pd.DataFrame(index=source)
        for name in list_station:
            s = sto[OP_type][name]
            coeff[name] = s.reg.params
            p[name]     = s.reg.pvalues
            bse[name]   = s.reg.bse
    
        print("\\midrule\n\\multicolumn{{{}}}{{c}}{{\\{}}}\\\\".format(len(list_station),OP_type))
        for i in range(len(source)):
            s = list()
            for name in list_station:
                # test if the source was in the PMF sources
                if coeff.index[i] in sto[OP_type][name].CHEM.columns:
                    if pd.isnull(coeff[name])[i]:
                        s.append(" --- ")
                    else:
                        s.append("{:.2f} $\\pm$ {:.2f} ({:.2f})".format(coeff[name][i],
                                                                        bse[name][i],
                                                                        p[name][i]))
                else:
                    s.append(" ")
                                                                            
            s = " & ".join(s)
            # s = re.sub("nan .{3,6} nan", " ", s)
            print(" {:8} & ".format(source[i].replace("_", " "))+ s + "\\\\") 
                # print(" & {:8} & ${:4.2f}x{:+3.2f}$ & {:3.2f} & {:3.2f} to {:3.2f}\\\\".format(s.name, s.p[0],s.p[1], s.r2[0,1],
                                       # s.OP.min(), s.OP.max()))
    print("\\bottomrule")

    print("\\toprule\nSource &" + " & ".join(list_station) +"\\\\")

    for OP_type in list_OPtype:
        p     = pd.DataFrame(index=source)
        for name in list_station:
            s = sto[OP_type][name]
            p[name]     = s.reg.pvalues
    
        print("\\midrule\n\\multicolumn{{{}}}{{l}}{{\\{}}}\\\\".format(len(list_station),OP_type))

        for i in range(len(source)):
            s = list()
            for name in list_station:
                # test if the source was in the PMF sources
                if p.index[i] in sto[OP_type][name].CHEM.columns:
                    if pd.isnull(p[name])[i]:
                        s.append(" --- ")
                    else:
                        s.append("{:.3f}".format(p[name][i]))
                else:
                    s.append(" ")
                                                                            
            s = " & ".join(s)
            # s = re.sub("nan .{3,6} nan", " ", s)
            print(" {:8} & ".format(source[i].replace("_", " "))+ s + "\\\\") 
                # print(" & {:8} & ${:4.2f}x{:+3.2f}$ & {:3.2f} & {:3.2f} to {:3.2f}\\\\".format(s.name, s.p[0],s.p[1], s.r2[0,1],
                                       # s.OP.min(), s.OP.max()))
    print("\\bottomrule")
