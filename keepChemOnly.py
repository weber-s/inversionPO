import os
import numpy as np
import datetime as dt
import pandas as pd


def to_date(strdate):
    # "31/10/2010 22:34"
    date=list()
    for d in strdate:
        if d=='':
            date.append('')
            #return float(-999)
        date.append(dt.datetime.strptime(d, '%d/%m/%Y'))
    date = np.array(date)
    return(date)


def extractData(station_list):
    print("Object created: \n---------------")
    for name in station_list:
        POfileConc = os.path.join(inputDir,name+'/'+name+"PO_conc.csv")
        POfileUnc = os.path.join(inputDir,name+'/'+name+"PO_unc.csv")
        CHEMfileConc = os.path.join(inputDir,name+'/'+name+"CHEM_conc.csv")
        CHEMfileUnc = os.path.join(inputDir,name+'/'+name+"CHEM_unc.csv")
        try:
            POconc[name]    = pd.read_csv(POfileConc, index_col=0)
            POunc[name]     = pd.read_csv(POfileUnc, index_col=0)
            CHEMconc[name]  = pd.read_csv(CHEMfileConc, index_col=0)
            CHEMunc[name]   = pd.read_csv(CHEMfileUnc, index_col=0)
            # re-order the DF to have timeserie in index
            POdate  = to_date(POconc[name].index)
            CHEMdate= to_date(CHEMconc[name].index)
            POconc[name].index  = POdate
            POunc[name].index   = POdate
            CHEMconc[name].index= CHEMdate
            CHEMunc[name].index = CHEMdate

        except FileNotFoundError:
            raise FileNotFoundError("{station}: the file {file} is not\
                                    found...".format(file=POfileConc, station=name))
            sys.exit()
        print(name)

def concatenateCHEMPO(station_list,keep,POconc,POunc,CHEMconc,CHEMunc):
    for name in station_list:
        print("\n===== {name} =====".format(name=name))
        datemin = min(min(POconc[name].index),min(CHEMconc[name].index))
        datemax = max(max(POconc[name].index),max(CHEMconc[name].index))
        print(datemin,datemax)
        ts      = pd.date_range(datemin,datemax)
        dfc      = pd.DataFrame(index=ts)
        dfu      = pd.DataFrame(index=ts)
        dfc.index.name = "date"
        dfu.index.name = "date"
        for key in keep:
            if key in CHEMconc[name].keys():
                dfc[key]=CHEMconc[name][key]
                dfu[key]=CHEMunc[name][key]
            elif key in POconc[name].keys():
                dfc[key]=POconc[name][key]
                dfu[key]=POunc[name][key]

        dfu = dfu.dropna()
        dfc = dfc.dropna()

        dfc.to_csv(outputDir+name+outputFile+"_conc.csv",na_rep=-999)
        dfu.to_csv(outputDir+name+outputFile+"_unc.csv",na_rep=-999)

# ========= PARAMETERS =======================================================
inputDir    = "/home/samuel/Documents/IGE/BdD_PO/"
outputDir   = "/home/samuel/Documents/IGE/PMF/DataPMF/"
outputFile  = "PMFclassique"

station_list= ("Fresnes",)
POconc, POunc = {}, {}
CHEMconc, CHEMunc = {}, {}



#keepChem=('OC','OC*','EC','MSA','Cl-','NO3-','SO42-','Na+','NH4+','K+','Mg2+','Ca2+','ΣPolyols','Levoglusan','As','Ba','Cd','Cr','Cu','Fe','Pb','Rb','Sb','Sn','Zn','Zr','ΣHAP_part','ΣAlc_wax','ΣBNT','ΣHOP','ΣMethoxy_part','PM10','PODTTµg','PODTTm3','POAAµg','POAAm3')
#keepChem=('OC','EC','MSA','Cl-','NO3-','SO42-','Na+','NH4+','K+','Mg2+','Ca2+','ΣPolyols','Levoglusan','Cu','Pb','Sn','Zn','ΣHAP_part','ΣAlc_wax','ΣMethoxy_part','PODTTµg','PODTTm3','POAAµg','POAAm3')

# PMF classique
CHEMconc.keys()
keep=('PM10','OC','OC*','EC','MSA','Cl-','NO3-','SO42-','Na+','NH4+','K+','Mg2+','Ca2+','ΣPolyols','Levoglucosan','As','Ba','Cd','Co','Cu','Fe','Mn','Mo','Ni','Pb','Rb','Sb','Sr','Ti','Zn','Zr','ΣHAP_part','ΣAlc_wax','ΣMethoxy_part','PODTTµg','PODTTm3','POAAµg','POAAm3')

extractData(station_list)

print(POconc["Fresnes"].index.name)
concatenateCHEMPO(station_list, keep, POconc,POunc,CHEMconc,CHEMunc)
