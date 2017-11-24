import pandas as pd
import numpy as np
from scipy import odr
import scipy.stats as stats

class Station:
    """
    This class is more a storage class for each station than a proper class...
    """
    def __init__(self, inputDir=None, name=None, SRCfile=None, OPfile=None,
                 CHEMfile=None,
                 SRC=None, CHEM=None, OP=None, OPunc=None, list_OPtype=None, OPi=None, OPiunc=None,
                 reg=None, modelunc=None,
                 ):

        self.inputDir = inputDir
        self.name   = name

        self.SRCfile = SRCfile
        self.OPfile = OPfile
        self.CHEMfile = CHEMfile

        self.list_OPtype = list_OPtype

        self.SRC    = SRC
        self.CHEM   = CHEM
        self.OP     = OP

        self.OPi    = pd.DataFrame(columns=list_OPtype)
        self.OPmodel_unc = pd.DataFrame(columns=list_OPtype) 
        self.reg    = {}

        self.OPmodel = pd.DataFrame(columns=list_OPtype)
        self.pearsonr = pd.DataFrame(columns=list_OPtype)

        self.odr = None
        self.odr_coeff = None

        #if OP is not None:
        #    self.model  = pd.Series((PM*OPi).sum(axis=1), name=name)
        #    self.odr    = self.get_ODR_result()
        #    self.p      = self.odr.beta
        #    self.p.shape= (2,)
        #    self.pearson_r = pd.concat([OP,self.model],axis=1).corr().as_matrix()
        #    self.covm   = covm
        #else:
        #    self.OPi    = pd.Series(index=PM.columns)
        #    self.model  = pd.Series(index=PM.index)
        #    self.pie    = pd.Series(index=PM.columns)
        #    self.p      = None
        #    self.pearson_r  = None
        #    self.covm   = pd.Series(index=PM.columns)
        #    self.yerr   = None
        #    self.reg    = None

        # self.covm.sort_index(inplace=True)
        # self.OPi.sort_index(inplace=True)
        
    def load_from_file(self, file):
        file = "{dir}/{station}/{station}{file}".format(dir=self.inputDir,
                                                         station=self.name,
                                                         file=file)
        df = pd.read_csv(file,
                         index_col=["date"], parse_dates=["date"],
                         na_values=['#VALUE!', '#DIV/0!','#VALEUR !'])
        return df

    def load_SRC(self):
        df = self.load_from_file(self.SRCfile)
        # remove <0 value
        df[df<0] = 0
        self.SRC = df

    def load_CHEM(self):
        df = self.load_from_file(self.CHEMfile)
        # if usecols == None:
        #     df[usecols,
        #TODO

        # warn <0 value
        if (df<0).any().any():
            print("WARNING: concentration negative in CHEM")
        self.CHEM = df

    def load_OP(self):
        df = self.load_from_file(self.OPfile)
        map_columns_name={"date prelevement": "date",
                          "PO_DTT_µg": "DTTm",
                          "SD_PO_DTT_µg": "SD_DTTm",
                          "PO_DTT_m3": "DTTv" ,
                          "SD_PO_DTT_m3": "SD_DTTv" ,
                          "PO_AA_µg": "AAm",
                          "SD_PO_AA_µg": "SD_AAm",
                          "PO_AA_m3": "AAv",
                          "SD_PO_AA_m3": "SD_AAv",
                          "nmol_[H202].µg-1": "DCFHm",
                          "SD_PO_DCFH_µg": "SD_DCFHm",
                          "nmol_[H202]equiv.m-3 air": "DCFHv",
                          "SD_PO_DCFH_m3": "SD_DCFHv"
                         }
        df.rename_axis(map_columns_name, axis="columns", inplace=True)

        colOK = [a for a in df.columns if a in
                 self.list_OPtype+",SD_".join(["nimp"]+self.list_OPtype).split(",")]
        df = df[colOK]
        df = df.dropna()
        # warn <0 value
        if (df<0).any().any():
            print("WARNING: concentration negative in OP")
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
            "VEH dir": "Vehicular_dir",
            "Oil/Vehicular": "Vehicular",
            "Road traffic": "Vehicular",
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
            "Marine secondary": "Secondary_bio",
            "Marin secondaire": "Secondary_bio",
            "HFO": "HFO",
            "Marin": "Secondary_bio",
            "Sea/road salt": "Salt",
            "Sea salt": "Salt",
            "Seasalt": "Salt",
            "Fresh seasalt": "Salt",
            "Sels de mer": "Salt",
            "Aged sea salt": "Aged_salt",
            "Aged seasalt": "Aged_salt",
            "Primary bio": "Primary_bio",
            "Primary biogenic": "Primary_bio",
            "Biogénique primaire": "Primary_bio",
            "Biogenique": "Primary_bio",
            "Biogenic": "Primary_bio",
            "Mineral dust": "Dust",
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



