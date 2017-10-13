import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from misc_utility.load_data import *

plt.interactive(True)

"""
Directory where the data are stored. Should be in the format
    {INPUT_DIR}/{stationName}/{stationName}
For instance:
    for the chemistry file:
        /home/foo/bar/BdD/Frenes/FrenesCHEM_conc.csv
    for the OP file:
        /home/foo/bar/BdD/Frenes/FrenesOP.csv
(the name of the file can be change in misc_utility/load_data.py in the
load_CHEMorOP function)

The files must be in CSV, with a column "date" in format ISO 8601 (YYYY-MM-DD).
The headers are the names of the species/sources or OP.

colOK   : the columns that will be used.

keep    : the OP names (column name of the OP file)

list_station : the names of the station to plot. It has to be the name of the
               folder where the CHEM and OP file are stored.
"""

INPUT_DIR = "/home/samuel/Documents/IGE/BdD/BdD_OP/"
colOK   = ("OCwb*","OCautres*","BCwb","BCff",\
           "Na+","NH4+","K+","Mg2+","Ca2+","Cl-","NO3-","SO42-",\
           "Levoglucosan","ΣPolyols","MSA",\
           "As","Cu","Fe","Mn","Mo","Ni","Pb","Rb","Sb","Ti","V","Zn","Zr",\
           # "Al","As","Ba","Cd","Cu","Fe","Mn","Mo","Ni","Pb","Sb","Sn","Ti","Zn",\
           # "Al","As","Ba","Cd","Ce","Co","Cr","Cu","Fe","La","Li","Mn","Mo","Ni","Pb","Rb","Sb","Se","Sn","Sr","Ti","Tl","V","Zn","Zr",\
           # "ΣHAP","ΣHOP","HULIS","DOC")
           "ΣHOP","Σmethoxy","PM10")
# colOK = ("Levoglucosan","ΣPolyols","MSA","Cu","Fe","Ox","NO3-","SO42-","ΣHOP","EC")
keep = ["DTTv","AAv"]

# list_station=["ANDRA","PdB","Marseille","Nice","Frenes","Chamonix","Marnaz","Passy"]
list_station=["Chamonix",]
# initialize the figure
f = plt.figure(figsize=(7.41,  5.59))
# set the color map + missing value in lightgrey
cmap = plt.get_cmap(name="RdBu_r")
cmap.set_bad(color='0.85')

# load the chemistry and OP, then plot the correlation matrix
# for each station 
for i, name in enumerate(list_station):
    CHEM    = load_CHEMorOP(name, INPUT_DIR, CHEMorOP="CHEM",
                            fromSource=False)
    # if "PM2.5" in CHEM.columns:
    #     CHEM = CHEM.div(CHEM["PM2.5"], axis="index")
    # else:
    #     CHEM = CHEM.div(CHEM["PM10"],axis="index")
    OP      = load_CHEMorOP(name, INPUT_DIR, CHEMorOP="OP")
    # keep only the days present in both CHEM and OP file
    TMP     = pd.merge(CHEM.ix[:,colOK], OP.ix[:,keep], how="inner",
                       right_index=True, left_index=True)
    # select only the raws of the OP, and skip the column of auto-correlation
    # OP/OP, i.e. a correlation of 1.
    corr    = TMP.corr().ix[keep,:-len(keep)]

    # plot the correlation matrix
    plt.subplot(len(list_station),1,i+1)
    im = plt.imshow(corr, cmap=cmap, vmin=-1, vmax=1)
    plt.ylabel(name)
    # plt.title(name)

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
f.subplots_adjust(top=1.0, bottom=0.16, left=0.12,right=0.85,hspace=0)
cbar_ax = f.add_axes([0.87, 0.20, 0.02, 0.76])
f.colorbar(im, cax=cbar_ax)

#
# ==================================================================
# ===== IDEM, BUT FOR THE SOURCES FROM THE PMF =====================
# ==================================================================
#

colOK = ['HFO', 'Marine/HFO', 'Aged_salt','Bio_burning', 'Dust', 'Industrial', 'Nitrate_rich', 'Primary_bio',
       'Salt', 'Secondary_bio', 'Sulfate_rich', 'Vehicular']
f = plt.figure(figsize=(7.41,  5.59))
cmap = plt.get_cmap(name="RdBu_r")
cmap.set_bad(color='0.85')
for i,name in enumerate(list_station):
    CHEM    = load_CHEMorOP(name, INPUT_DIR, CHEMorOP="CHEM", fromSource=True)
    CHEM = setSourcesCategories(CHEM)
    to_merge =["Bio_burning", "Bio_burning1","Bio_burning2",\
               "Vehicular", "Vehicular_ind","Vehicular_dir"]
    for j in range(0,int(len(to_merge)/3)+3,3):
        try:
            CHEM[to_merge[j]] = CHEM[to_merge[j+1:j+3]].sum(axis=1)
            CHEM.drop(to_merge[j+1:j+3], axis=1,inplace=True)
        except:
            pass
            # print("The sources {merge} are not in the site {site}".format(merge=to_merge[j+1:j+3], site=name))
    CHEM.sort_index(axis=1, inplace=True)
    OP      = load_CHEMorOP(name, INPUT_DIR, CHEMorOP="OP")
    # TMP     = pd.concat([CHEM.ix[:,colOK],OP.ix[:,keep]],axis=1, join="inner")
    TMP     = pd.merge(CHEM.ix[:,colOK], OP.ix[:,keep], how="inner",
                       right_index=True, left_index=True)
    corr    = TMP.corr().ix[keep,:-len(keep)]

    # plt.subplot(1,len(list_station),i+1)
    plt.subplot(len(list_station),1,i+1)
    im = plt.imshow(corr, cmap=cmap, vmin=-1, vmax=1)
    plt.ylabel(name)
    # plt.title(name)

f.subplots_adjust(wspace=0)
a = f.axes[-1];
a.set_xticks(range(len(corr.columns)))
a.set_xticklabels([l.replace("_"," ") for l in corr.columns], rotation=90)
[a.set_xticks([]) for a in f.axes[:-1]];
[a.set_yticks(range(len(corr.index))) for a in f.axes]
[a.set_yticklabels(corr.index, rotation=00) for a in f.axes]

f.subplots_adjust(top=0.97, bottom=0.20, left=0.15,right=0.85,hspace=0.20)
cbar_ax = f.add_axes([0.80, 0.25, 0.02, 0.67])
f.colorbar(im, cax=cbar_ax)
