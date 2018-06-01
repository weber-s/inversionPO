import pandas as pd
import numpy as np
from scipy import odr
import scipy.stats as stats
import statsmodels.api as sm

class Station:
    """
    This class is more a storage class for each station than a proper class...
    """
    def __init__(self, name=None, SRC=None, OP=None, OPunc=None, 
                 list_OPtype=None, OPi=None, OPiunc=None, 
                 reg=None, modelunc=None,
                 ):

        self.name   = name

        self.list_OPtype = list_OPtype

        self.SRC    = SRC
        self.OP     = OP

        self.OPi    = pd.DataFrame(columns=list_OPtype)
        self.OPmodel_unc = pd.DataFrame(columns=list_OPtype) 
        self.reg    = {}

        self.OPmodel = pd.DataFrame(columns=list_OPtype)
        self.pearsonr = pd.DataFrame(columns=list_OPtype)

        self.odr = None
        self.odr_coeff = None

    def load_from_file(self, file, overwriteBDIR=True):
        if not overwriteBDIR:
            file = "{dir}/{station}/{station}{file}".format(dir=self.inputDir,
                                                            station=self.name,
                                                            file=file)
        else:
            file = file

        df = pd.read_csv(file,
                         index_col=["date"], parse_dates=["date"],
                         na_values=['#VALUE!', '#DIV/0!','#VALEUR !'])
        return df

    def load_SRC(self, df=None):
        to_exclude = ["index", "station", "programme"]
        df.dropna(axis=1, how="all", inplace=True)
        df.drop(to_exclude, axis=1, inplace=True)
        df.date = pd.to_datetime(df.date)
        df.set_index(["date"], drop=True, inplace=True)
        # remove <0 value
        # df[df<0] = 0
        self.SRC = df.sort_index(axis=1)

    def load_CHEM(self, overwriteBDIR=True):
        df = self.load_from_file(self.CHEMfile, overwriteBDIR)
        # if usecols == None:
        #     df[usecols,
        #TODO

        # warn <0 value
        if (df<0).any().any():
            print("WARNING: concentration negative in CHEM")
        self.CHEM = df

    def load_OP(self, df=None):
        map_columns_name={"date prelevement": "date",
                          "PO_DTT_µg": "DTTm",
                          "SD_PO_DTT_µg": "SD_DTTm",
                          "PO_DTT_m3": "DTTv" ,
                          "SD_PO_DTT_m3": "SD_DTTv" ,
                          "PO_AA_µg": "AAm",
                          "SD_PO_AA_µg": "SD_AAm",
                          "PO_AA_m3": "AAv",
                          "SD_PO_AA_m3": "SD_AAv",
                          "nmol_H2O2.µg-1": "DCFHm",
                          "SD_PO_DCFH_µg": "SD_DCFHm",
                          "nmol_H2O2equiv.m-3 air": "DCFHv",
                          "SD_PO_DCFH_m3": "SD_DCFHv"
                         }
        df.rename(map_columns_name, axis="columns", inplace=True)
        df.date = pd.to_datetime(df.date)
        df.set_index("date", drop=True, inplace=True)

        colOK = []
        for OPtype in self.list_OPtype:
            colOK += [var for var in df.columns if OPtype in var]
        df = df[colOK]
        df = df.dropna(how='all')
        # warn <0 value
        if (df<0).any().any():
            print("WARNING: concentration negative in OP")
        if any(df==0):
            print("WARNING: some value are 0 in OP or SD OP, set it to 1e-5")
            df[df==0] = 1e-5
        self.OP = df

    def setSourcesCategories(self):
        """
        Return the DataFrame with renamed columns
        The renaming set the source's name to its category, i.e
        Road traffic → Vehicular
        VEH → Vehicular
        Secondary bio → Secondary_bio
        BB → Bio_burning
        Biomass burning → Bio_burning
        etc.
        """
        possible_sources ={
            "Vehicular": "Vehicular",
            "VEH": "Vehicular",
            "VEH ind": "Vehicular_ind",
            "Traffic_exhaust": "Traffic_exhaust",
            "Traffic_non-exhaust": "Traffic_non-exhaust",
            "VEH dir": "Vehicular_dir",
            "Oil/Vehicular": "Vehicular",
            "Road traffic": "Road dust",
            "Road trafic": "Vehicular",
            "Bio. burning": "Bio_burning",
            "Bio burning": "Bio_burning",
            "Comb fossile/biomasse": "Bio_burning",
            "BB": "Bio_burning",
            "Biomass Burning": "Bio_burning",
            "Biomass burning": "Bio_burning",
            "BB1": "Bio_burning1",
            "BB2": "Bio_burning2",
            "Sulfate-rich": "Sulfate_rich",
            "Nitrate-rich": "Nitrate_rich",
            "Sulfate rich": "Sulfate_rich",
            "Nitrate rich": "Nitrate_rich",
            "Secondaire": "Secondary_bio",
            "Secondary bio": "Secondary_bio",
            "Secondary biogenic": "Secondary_bio",
            "Secondaire organique": "Secondary_bio",
            "Marine biogenic/HFO": "Marine/HFO",
            "Secondary biogenic/HFO": "Marine/HFO",
            "Marine bio/HFO": "Marine/HFO",
            "Marin bio/HFO": "Marine/HFO",
            "Sulfate rich/HFO": "Marine/HFO",
            "Marine secondary": "Secondary_bio",
            "Marin secondaire": "Secondary_bio",
            "HFO": "HFO",
            "Marin": "Secondary_bio",
            "Sea/road salt": "Salt",
            "Road salt": "Salt",
            "Sea salt": "Salt",
            "Seasalt": "Salt",
            "Fresh seasalt": "Salt",
            "Sels de mer": "Salt",
            "Aged sea salt": "Aged_salt",
            "Aged seasalt": "Aged_salt",
            "Aged seasalt": "Aged_salt",
            "Primary bio": "Primary_bio",
            "Primary biogenic": "Primary_bio",
            "Biogénique primaire": "Primary_bio",
            "Biogenique": "Primary_bio",
            "Biogenic": "Primary_bio",
            "Mineral dust": "Dust",
            "Mineral dust ": "Dust",
            "Resuspended dust": "Dust",
            "Dust": "Dust",
            "Crustal dust": "Dust",
            "Dust (mineral)": "Dust",
            "Dust/biogénique marin": "Dust",
            "AOS/dust": "Dust",
            "Industrial": "Industrial",
            "Industrie": "Industrial",
            "Industries": "Industrial",
            "Industry/vehicular": "Indus/veh",
            "Industries/trafic": "Vehicular",
            "Fioul lourd": "Industrial",
            "Arcellor": "Industrial",
            "Siderurgie": "Industrial",
            "Débris végétaux": "Plant_debris",
            "Chlorure": "Chloride",
            "PM other": "Other"
            }
        self.SRC.rename(columns=possible_sources, inplace=True)
        self.SRC.sort_index(axis=1, inplace=True)
    
    def mergeSources(self, inplace=True):
        """
        Merge the different Biomass burning and Vehicular sources into a single one.

        Parameters:
        -----------
        
        self.SRC: pandas DataFrame
        inplace: boolean, default True. Change the sources in self.SRC.

        Output:
        -------

        df: pandas DataFrame with sources merged.

        """
        
        df_merged= self.SRC.copy()

        to_merge =[["Bio_burning", "Bio_burning1","Bio_burning2"],\
                   ["Vehicular", "Vehicular_ind","Vehicular_dir"]]

        for source in to_merge:
            if (source[1] in self.SRC.columns or source[2] in self.SRC.columns): 
                df_merged[source[0]] = df_merged[source[1:]].sum(axis=1)
                df_merged.drop(source[1:], axis=1,inplace=True)

        if inplace:
            self.SRC = df_merged
        else:
            return df_merged

    def get_WLS_result(self, OPtype, x=None, y=None, sy=None):
        if x == None:
            x = self.OP[OPtype]
            y = self.reg[OPtype].fittedvalues
            idx = x.index.intersection(y.index)
                          
            mywls = sm.WLS(y[idx],
                          sm.add_constant(x[idx]),
                          weights=1/self.OPmodel_unc.loc[idx, OPtype])
        else:
            mywls = sm.WLS(y, x, sy)
        output = mywls.fit()
        self.wls = mywls
        self.wls_coeff = output.params
        return output

    def get_ODR_result(self, OPtype, x=None, y=None, sx=None, sy=None):
        if x == None:
            idx = self.OP[OPtype].index.intersection(self.OPmodel[OPtype].index)
            data = odr.RealData(self.OP.loc[idx, OPtype],
                                self.OPmodel.loc[idx, OPtype],
                                sx=self.OP.loc[idx, "SD_"+OPtype],
                                sy=self.OPmodel_unc.loc[idx, OPtype])
        else:
            data = odr.RealData(x, y, sx, sy)
        myodr  = odr.ODR(data, odr.unilinear)
        output = myodr.run()
        self.odr = myodr
        self.odr_coeff = output.beta
        return output.beta

    def get_pearson_r(self, OPtype):
        idx = self.OP[OPtype].index.intersection(self.OPmodel[OPtype].index)
        pearson = stats.pearsonr(self.OP.loc[idx, OPtype],
                                 self.OPmodel.loc[idx, OPtype])
        self.pearsonr[OPtype] = pearson
        return pearson



