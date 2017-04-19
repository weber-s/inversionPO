import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import pickle
from scipy import linalg
from scipy import polyfit

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
        "Marseille": "#e377c2"
    }
    color = pd.DataFrame(index=["color"], data=color)
    return color

def sourcesColor():
    color ={
        "Vehicular": "#000000",
        "Vehicular": "#000000",
        "VEH": "#000000",
        "VEH ind": "#111111",
        "Vehicular_ind": "#111111",
        "Vehicular ind": "#111111",
        "VEH dir": "#333333",
        "Vehicular_dir": "#333333",
        "Vehicular dir": "#333333",
        "Oil/Vehicular": "#000000",
        "Road traffic": "#000000",
        "Bio_burning": "#92d050",
        "Bio_burning1": "#92d050",
        "Bio burning1": "#92d050",
        "Bio_burning2": "#bbd020",
        "Bio burning2": "#bbd020",
        "Bio. burning": "#92d050",
        "Bio burning": "#92d050",
        "BB": "#92d050",
        "BB1": "#92d050",
        "BB2": "#bbd020",
        "Sulfate_rich": "#ff2a2a",
        "Sulfate-rich": "#ff2a2a",
        "Sulfate rich": "#ff2a2a",
        "Nitrate_rich": "#ff7f2a",
        "Nitrate-rich": "#ff7f2a",
        "Nitrate rich": "#ff7f2a",
        "Secondaire": "#ff5f2a",
        "Secondary_bio": "#8c564b",
        "Secondary bio": "#8c564b",
        "Secondary biogenic": "#8c564b",
        "Marine_bio/HFO": "#8c564b",
        "Marine biogenic/HFO": "#8c564b",
        "Marine bio/HFO": "#8c564b",
        "Marin bio/HFO": "#8c564b",
        "Marine_bio": "#fc564b",
        "Marine bio": "#fc564b",
        "Marine secondary": "#fc564b",
        "Marin secondaire": "#fc564b",
        "HFO": "#70564b",
        "Marine": "#33b0f6",
        "Marin": "#33b0f6",
        "Salt": "#00b0f0",
        "Sea/road salt": "#00b0f0",
        "Sea salt": "#00b0f0",
        "Aged_salt": "#00b0ff",
        "Aged salt": "#00b0ff",
        "Aged sea salt": "#00b0ff",
        "Aged seasalt": "#00b0ff",
        "Primary_bio": "#ffc000",
        "Primary bio": "#ffc000",
        "Primary biogenic": "#ffc000",
        "Biogenique": "#ffc000",
        "Biogenic": "#ffc000",
        "Dust": "#dac6a2",
        "Dust (mineral)": "#dac6a2",
        "Mineral dust": "#dac6a2",
        "Resuspended dust": "#dac6a2",
        "AOS/dust": "#dac6a2",
        "Industrial": "#7030a0",
        "Indus._veh.": "#7030a0",
        "Industry/vehicular": "#7030a0",
        "Arcellor": "#7030a0",
        "Siderurgie": "#7030a0",
        "Plant_debris": "#2aff80",
        "Débris végétaux": "#2aff80",
        "Choride": "#80e5ff",
        "Chlorure": "#80e5ff",
        "Other": "#cccccc",
        "PM other": "#cccccc",
        "nan": "#ffffff"
    }
    color = pd.DataFrame(index=["color"], data=color)
    return color

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
    pd.concat((obs,model), axis=1).plot(ax=ax,x=[0], y=[1], kind="scatter")
    plt.plot([0,obs.max()],[0, obs.max()], '--', label="y=x")
    plt.plot([0,obs.max()],[p[1], p[0]*obs.max()+p[1]], label="linear fit")
    posy = 0.7*plt.ylim()[1]
    plt.text(0,posy,"y=%.2fx+%0.2f\nr²=%0.2f" % (p[0],p[1],r2[0,1]))
    plt.xlabel("Obs.")
    plt.ylabel("Recons.")
    plt.title("obs. vs reconstruction")
    ax.set_aspect(1./ax.get_data_ratio())
    l=plt.legend(loc="lower right")
    l.draw_frame(False)

def plot_station_sources(station,**kwarg):
    """
    Plot the mass contrib (piechart), the scatter plot obs/recons, the
    intrinsic PO and the contribution of the sources/species (TS + piechart).
    TODO: it's ugly...
    """
    plt.figure(figsize=(17,8))
    # Mass contribution (pie chart)
    ax=plt.subplot(2,3,1)
    plot_contribPie(ax, station.pieCHEM)
    # Bar plot of coeff for the PO
    ax=plt.subplot(2,3,2)
    plot_coeff(station,ax)
    plt.ylabel("PO [nmol/min/µg]")
    # Scatter plot obs/recons.
    ax=plt.subplot(2,3,3)
    plot_scatterReconsObs(ax, station.PO, station.model, station.p, station.r2)
    # time serie reconstruction/observation
    ax=plt.subplot(2,3,(4,5))
    plot_ts_reconstruction_PO(station,ax=ax)
    plt.legend(mode="expand", bbox_to_anchor=(0.5,-0.1))
    # PO contribution (pie chart)
    ax=plt.subplot(2,3,6)
    plot_contribPie(ax, station)

    plt.subplots_adjust(top=0.95, bottom=0.16, left=0.07, right=0.93)

def plot_station(station,POtype,**kwarg):
    """
    Plot the time series obs & recons, the scatter plot obs/recons, the
    intrinsic PO and the contribution of the sources/species.
    """
    plt.figure(figsize=(17,8))
    # time serie reconstruction/observation
    ax=plt.subplot(2,3,(1,3))
    ax.errorbar(station.PO.index.to_pydatetime(), station.PO,
                yerr=station.POunc, 
                ecolor="black",
                elinewidth=1,
                fmt="b-o",
                markersize=6,
                label="Obs.",
                zorder=1)
    ax.plot_date(station.model.index.to_pydatetime(), station.model, "r-*",
                 label="Recons.",zorder=10)
    ax.set_ylabel("{PO} loss\n[nmol/min/m³]".format(PO=POtype[:-1]))
    plt.title("{station} {POt}".format(station=station.name, POt=POtype))
    handles, labels = ax.get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    l=ax.legend(handles, labels) # in order to have Obs 1st
    l.draw_frame(False)
    # scatter plot reconstruction/observation
    ax=plt.subplot(2,3,4)
    plot_scatterReconsObs(ax, station.PO, station.model, station.p, station.r2)
    # factors contribution
    ax=plt.subplot(2,3,5)
    plot_coeff(station, ax=ax)
    plt.ylabel("PO [nmol/min/µg]")
    # Pie chart
    ax=plt.subplot(2,3,6)
    plot_contribPie(ax, station,**kwarg)

    plt.subplots_adjust(top=0.95, bottom=0.16, left=0.07, right=0.93)

def plot_contribPie(ax, station, fromSource=True, title=None, ylabel=None,
                    **kwarg):
    """
    Plot contributions of the sources to the PO in a Pie chart
    The contributions is G*m.
    """
    # check if station is an object or a DataFrame
    if isinstance(station, pd.Series):
        df = station
    else:
        if not(station.hasPO):
            ax.set_aspect('equal')
            p = station.pie.plot.pie(ax=ax, **kwarg)
            ax.set_ylabel("")
            return
        df = station.pie
    
    l = df.index
    l = [a.replace("_"," ") for a in l]
    df.index = l

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
    print(p)

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
    """Plot a bar plot of the intrinsique PO of the sources for all the station"""
    if ax==None:
        ax = plt.subplots(row=len(POtype_list), columns=len(stations.keys()))

    c = sitesColor()
    cols = list()
    try:
        for s in stations:
            cols.append(c.ix["color"][s])
        stations.plot.bar(ax=ax, yerr=yerr, legend=False, color=cols,rot=30)
    except TypeError:
        cols.append(c.ix["color"][stations.name])
        stations.m.plot.bar(ax=ax, yerr=stations.covm, legend=False, color=cols)


def plot_ts_contribution_PO(station,POtype=None,saveDir=None):
    """
    Plot the time serie contribution of each source to the PO.
    station can be the name of the station or a Station object.
    If station is a string, then the saveDir variable must be the path to the
    directory where the file is saved.
    The file name must be in the format 
        {station name}_contribution_{POtype}.csv
    """
    
    if isinstance(station, str):
        if saveDir == None:
            print("ERROR: the 'saveDir' argument must be completed")
            return
        print("Use the saved results")
        title = station 
        fileName = saveDir+station+"_contribution_"+POtype+".csv"
        df = pd.read_csv(fileName,index_col="date", parse_dates=["date"])
    else:
        df = station.CHEM * station.m
        title = station.name

    c = sourcesColor()
    cols = c.ix["color",df.columns].values
    
    df.plot(title=title, color=cols)
    plt.ylabel(POtype)
    return

def plot_ts_reconstruction_PO(station, POtype=None, POobs=None, saveDir=None, ax=None):
    """
    Plot a stacked barplot of for the sources contributions to the PO
    """
    if ax == None:
        f, ax = plt.subplots(1, figsize=(10,5))

    if isinstance(station, str):
        if saveDir == None or POobs == None:
            print("ERROR: the 'saveDir' and 'POobs' arguments must be completed")
            return
        title = station 
        fileName = saveDir+station+"_contribution_"+POtype+".csv"
        PO = POobs
        df = pd.read_csv(fileName,index_col="date", parse_dates=["date"])
    else:
        df = station.CHEM * station.m
        PO = station.PO.values
        POunc = station.POunc.values
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
    # PO observation
    ax.errorbar(x, PO, POunc, fmt='ob', ecolor="black", elinewidth=1, markersize=3, label="OP obs.")

    # legend stuff
    ncol = int((len(df.columns)+1)/2)
    nrow = (len(df.columns)+1)/ncol
    if nrow > 2:
        ncol += 1
    plt.legend(loc="center",ncol=ncol,bbox_to_anchor=(0.5,-0.16))
    plt.title(title)
    plt.ylabel(POtype)

    plt.subplots_adjust(top=0.90, bottom=0.20, left=0.10, right=0.90)
    return

def plot_seasonal_contribution(station, POtype=None, saveDir=None,**kwarg):
    """
    Plot a stacked bar plot of the normalized contribution of the source to the
    PO.
    """

    # first, check of station is a string
    # if so, then load the associated Station class (previously saved).
    if isinstance(station, str):
        with open(saveDir+"/"+station+"_"+POtype+".pickle","rb") as f:
            station = pickle.load(f)
    
    df = station.m * station.CHEM

    add_season(df)

    df_grouped = df.groupby("season").sum()
    ordered_season = ["DJF","MAM","JJA","SON"]
    df_grouped = df_grouped.reindex(ordered_season)

    # selection the colors we have in the sources
    colors  = sourcesColor()
    c       = colors.ix["color", df_grouped.columns]
    # plot the stacked normalized bar plot
    axes = (df_grouped.T / df_grouped.sum(axis=1)).T.plot.bar(stacked=True,
                                                              rot=0,
                                                              color=c,
                                                              **kwarg)
    ax = plt.gca()
    ax.legend(loc="center",ncol=round(len(df_grouped.columns)/2), bbox_to_anchor=(0.5,-0.2))
    ax.set_ylabel("PO contribution (normalized)")
    plt.title(station.name+" (DTTv)")
    plt.subplots_adjust(top=0.90, bottom=0.20, left=0.15, right=0.85)

def plot_seasonal_contribution_boxplot(station, POtype=None, saveDir=None,**kwarg):
    """
    Plot a boxplot contribution of the source to the PO per season.
    """

    # first, check of station is a string
    # if so, then load the associated Station class (previously saved).
    if isinstance(station, str):
        with open(saveDir+"/"+station+"_"+POtype+".pickle","rb") as f:
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
    df_long = pd.melt(df,"season",var_name="source", value_name="PO")
    ax = sns.boxplot("season", y="PO",hue="source",data=df_long,palette=c)

    if "title" in kwarg:
        plt.title(kwarg["title"])
