import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import linalg
from scipy import polyfit


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
        "Nice": "#9467bd"
    }
    color = pd.DataFrame(index=["color"], data=color)
    return color

def sourcesColor():
    color ={
        "Vehicular": "#000000",
        "Oil/Vehicular": "#000000",
        "Road traffic": "#000000",
        "Bio. burning": "#92d050",
        "Bio burning": "#92d050",
        "BB": "#92d050",
        "Sulfate-rich": "#ff2a2a",
        "Nitrate-rich": "#ff7f2a",
        "Secondary bio": "#8c564b",
        "Marine biogenic/HFO": "#8c564b",
        "Marine bio/HFO": "#8c564b",
        "Sea/road salt": "#00b0f0",
        "Sea salt": "#00b0f0",
        "Aged sea salt": "#00b0ff",
        "Primary bio": "#ffc000",
        "Mineral dust": "#dac6a2",
        "Resuspended dust": "#dac6a2",
        "Dust": "#dac6a2",
        "AOS/dust": "#dac6a2",
        "Industrial": "#7030a0",
        "Industry/vehicular": "#7030a0",
        "Débris végétaux": "#2aff80",
        "Chlorure": "#80e5ff",
        "PM other": "#cccccc",
        "nan": "#ffffff"
    }
    color = pd.DataFrame(index=["color"], data=color)
    return color

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

def plot_station_sources(station):
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

def plot_station(station,POtype):
    """
    Plot the time series obs & recons, the scatter plot obs/recons, the
    intrinsic PO and the contribution of the sources/species.
    """
    plt.figure(figsize=(17,8))
    # time serie reconstruction/observation
    ax=plt.subplot(2,3,(1,3))
    ax.plot_date(station.PO.index.to_pydatetime(), station.PO, "b-o", label="Obs.")
    ax.plot_date(station.model.index.to_pydatetime(), station.model, "r-*", label="recons.")
    plt.title("{station} {POt}".format(station=station.name, POt=POtype))
    l=ax.legend(('Obs.','Recons.'))
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
    plot_contribPie(ax, station)

    plt.subplots_adjust(top=0.95, bottom=0.16, left=0.07, right=0.93)

def plot_contribPie(ax, station, fromSource=True, title=None, ylabel=None):
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
            station.pie.plot.pie(ax=ax)
            ax.set_ylabel("")
            return
        df = station.pie

    if fromSource:
        c = sourcesColor()
        cols = c.ix["color",df.index].values
        ax.set_aspect('equal')
        df.plot.pie(ax=ax,
                    shadow=False,
                    startangle=90,
                    colors=cols)
    else:
        ax.set_aspect('equal')
        df.plot.pie(ax=ax,
                    shadow=False,
                    startangle=90)

    if title is not None:
        ax.set_title(title)
    ax.set_ylabel("")

def plot_coeff(stations, ax=None):
    """Plot a bar plot of the intrinsique PO of the sources for all the station"""
    if ax==None:
        ax = plt.subplots(row=len(POtype_list), columns=len(stations.keys()))

    c = sitesColor()
    cols = list()
    try:
        for s in stations:
            cols.append(c.ix["color"][s])
        stations.plot.bar(ax=ax, legend=False, color=cols,rot=30)
    except TypeError:
        cols.append(c.ix["color"][stations.name])
        stations.m.plot.bar(ax=ax, legend=False, color=cols)


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

    count = 0
    for i in range(df.shape[1]):
        bottom=df.ix[:,0:count].sum(axis=1)
        count += 1
        ax.bar(x, df[df.columns[i]],
               bottom=bottom,
               label=df.columns[i],
               width=width,
               color=c[df.columns[i]])
    ax.plot(x, PO, 'xr', label="OP obs.")
    ncol = int((len(df.columns)+1)/2)
    nrow = (len(df.columns)+1)/ncol
    if nrow > 2:
        ncol += 1
    plt.legend(loc="center",ncol=ncol,bbox_to_anchor=(0.5,-0.16))
    plt.title(title)
    plt.ylabel(POtype)

    plt.subplots_adjust(top=0.90, bottom=0.20, left=0.10, right=0.90)
    return



