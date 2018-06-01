import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sqlite3

plt.interactive(True)
sns.set_style("white", {"axes.linewidth": 0})


"""
Directory where the data are stored. Should be in the format
    {INPUT_DIR}/{stationName}/{stationName}_PM.csv
For instance:
        /home/foo/bar/BdD/GRE-fr/GRE-fr_PM.csv

The files must be in CSV, with a column "date" in format ISO 8601 (YYYY-MM-DD).
The headers are the names of the species or sources.

colOK   : the columns that will be used from the CSV file.

keep    : the columns to compare with.

list_station : the names of the station to plot.
"""

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

# ========================================================================
# Parametres (to adapt)
# ========================================================================

INPUT_DIR = "/home/webersa/Documents/BdD/BdD_PM/sites/"
# names of columns to compare with
colOK   = [
    "PM10",
    "OC","EC",\
    "Na+","NH4+","K+","Mg2+","Ca2+","Cl-","NO3-","SO42-",\
    "Levoglucosan","Mannosan",
    "Arabitol","Mannitol","Sorbitol","Glucose","MSA",\
    # "As","Cu","Fe","Mn","Mo","Ni","Pb","Rb","Sb","Ti","V","Zn","Zr",\
    # "Al","As","Ba","Cd","Cu","Fe","Mn","Mo","Ni","Pb","Sb","Sn","Ti","Zn",\
    "Al","As","Ba","Cd","Ce","Co","Cr","Cu","Fe","La","Li","Mn","Mo","Ni","Pb","Rb","Sb","Se","Sn","Sr","Ti","Tl","V","Zn","Zr",\
    # "ΣHAP","ΣHOP","HULIS","DOC",
    "PAH Sum", "PAHs Sum", "Hopane Sum",
    "PO_DTT_m3", "PO_AA_m3", "nmol_H2O2equiv.m-3 air"
]
# colOK = ("Levoglucosan","ΣPolyols","MSA","Cu","Fe","Ox","NO3-","SO42-","ΣHOP","EC")

# species for comparison
keep = ["PO_DTT_m3","PO_AA_m3","nmol_H2O2equiv.m-3 air"]
# keep = ["Polyols","Arabitol","Mannitol","Sorbitol"]

# List of stations to plot
# list_station=["ANDRA","PdB","Marseille","Nice","Frenes","Chamonix","Marnaz","Passy"]
list_station=["Pipiripi", "El Alto"]#ANDRA-PM10","PdB","MRS-5av","Nice","GRE-fr","Chamonix","Marnaz","Passy"]

# ========================================================================
# Plot part
# ========================================================================

conn = sqlite3.connect("/home/webersa/Documents/BdD/BdD_PM/db.sqlite")   
for season in ["La Paz Experiment"]:#"DJF", "MAM", "JJA", "SON"]:
    # initialize the figure
    f = plt.figure(figsize=(9.41,  5.59))
    # set the color map + missing value in lightgrey
    cmap = plt.get_cmap(name="RdBu_r")
    cmap.set_bad(color='0.85')

    # load the chemistry and OP, then plot the correlation matrix
    # for each station 
    for i, name in enumerate(list_station):
        # station_file = INPUT_DIR+"/"+name+"/"+name+"_PM.csv"
        # df = pd.read_csv(station_file, index_col=["date"], parse_dates=True)
        df = pd.read_sql("SELECT * FROM values_all WHERE station in ('{station}');".format(station=name), 
                         con=conn,
                         index_col=["date"],
                         parse_dates=["date"])
        colKO = set(df.columns) - set(colOK) 
        df.drop(colKO, axis=1, inplace=True)
        for c in colOK:
            if c not in df.columns:
                df[c] = np.nan
        # if "Polyols" not in df.columns:
        #     df["Polyols"] = df[["Arabitol","Mannitol","Sorbitol"]].sum(axis=1)
        # add_season(df)
        # corr    = df[df["season"]==season].corr().ix[keep,colOK]
        corr    = df.corr().ix[keep,colOK]
        corr.rename({"nmol_H2O2equiv.m-3 air": "PO_DCFH_m3"}, axis=1, inplace=True)
        corr.rename({"nmol_H2O2equiv.m-3 air": "PO_DCFH_m3"}, axis=0, inplace=True)
        
        # plot the correlation matrix
        plt.subplot(len(list_station),1,i+1)
        im = plt.imshow(corr, cmap=cmap, vmin=-1, vmax=1)
        # im = sns.heatmap(corr, cmap=cmap, vmin=-1, vmax=1, cbar=False, annot=True, fmt=".2f")
        plt.title("{name} (n={n})".format(name=name,n=df.shape[0]))
        # plt.ylabel("{name} (n={n})".format(name=name,n=df.shape[0]))
    f.suptitle(season)

    # adjust the figure layout (delete some ticks, add the names of the columns,
    # etc.)
    f.subplots_adjust(wspace=0)
    a = f.axes[-1];
    a.set_xticks(range(len(corr.columns)))
    a.set_xticklabels(corr.columns, rotation=90)
    [a.set_xticks([]) for a in f.axes[:-1]];
    [a.set_yticks(range(len(corr.index))) for a in f.axes]
    [a.set_yticklabels(corr.index, rotation=00) for a in f.axes]

    # add the colorbar in the plot
    top = .9
    bottom = .16
    f.subplots_adjust(top=top, bottom=bottom, left=0.12,right=0.85,hspace=0.05)
    cbar_ax = f.add_axes([0.87, bottom+0.04, 0.02, top-bottom-0.04*2])
    f.colorbar(im, cax=cbar_ax)

#
# ==================================================================
# ===== IDEM, BUT FOR THE SOURCES FROM THE PMF =====================
# ==================================================================
#

# colOK = ['HFO', 'Marine/HFO', 'Aged_salt','Bio_burning', 'Dust', 'Industrial', 'Nitrate_rich', 'Primary_bio',
#        'Salt', 'Secondary_bio', 'Sulfate_rich', 'Vehicular']
# f = plt.figure(figsize=(7.41,  5.59))
# cmap = plt.get_cmap(name="RdBu_r")
# cmap.set_bad(color='0.85')
# for i,name in enumerate(list_station):
#     CHEM    = load_CHEMorOP(name, INPUT_DIR, CHEMorOP="CHEM", fromSource=True)
#     CHEM = setSourcesCategories(CHEM)
#     to_merge =["Bio_burning", "Bio_burning1","Bio_burning2",\
#                "Vehicular", "Vehicular_ind","Vehicular_dir"]
#     for j in range(0,int(len(to_merge)/3)+3,3):
#         try:
#             CHEM[to_merge[j]] = CHEM[to_merge[j+1:j+3]].sum(axis=1)
#             CHEM.drop(to_merge[j+1:j+3], axis=1,inplace=True)
#         except:
#             pass
#             # print("The sources {merge} are not in the site {site}".format(merge=to_merge[j+1:j+3], site=name))
#     CHEM.sort_index(axis=1, inplace=True)
#     OP      = load_CHEMorOP(name, INPUT_DIR, CHEMorOP="OP")
#     # TMP     = pd.concat([CHEM.ix[:,colOK],OP.ix[:,keep]],axis=1, join="inner")
#     TMP     = pd.merge(CHEM.ix[:,colOK], OP.ix[:,keep], how="inner",
#                        right_index=True, left_index=True)
#     corr    = TMP.corr().ix[keep,:-len(keep)]
#
#     # plt.subplot(1,len(list_station),i+1)
#     plt.subplot(len(list_station),1,i+1)
#     im = plt.imshow(corr, cmap=cmap, vmin=-1, vmax=1)
#     plt.ylabel(name)
#     # plt.title(name)
#
# f.subplots_adjust(wspace=0)
# a = f.axes[-1];
# a.set_xticks(range(len(corr.columns)))
# a.set_xticklabels([l.replace("_"," ") for l in corr.columns], rotation=90)
# [a.set_xticks([]) for a in f.axes[:-1]];
# [a.set_yticks(range(len(corr.index))) for a in f.axes]
# [a.set_yticklabels(corr.index, rotation=00) for a in f.axes]
#
# f.subplots_adjust(top=0.97, bottom=0.20, left=0.15,right=0.85,hspace=0.20)
# cbar_ax = f.add_axes([0.80, 0.25, 0.02, 0.67])
# f.colorbar(im, cax=cbar_ax)
