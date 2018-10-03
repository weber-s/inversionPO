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

        self.name = name

        self.list_OPtype = list_OPtype

        self.SRC = SRC
        self.OP = OP

        self.OPi = pd.DataFrame(columns=list_OPtype)
        self.OPmodel_unc = pd.DataFrame(columns=list_OPtype)
        self.reg = {}

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
                         na_values=['#VALUE!', '#DIV/0!', '#VALEUR !'])
        return df

    def load_SRC(self, dfcontrib=None, dfprofile=None, conn=None, programme=None):
        if conn:
            sql = "SELECT * from {{table}} WHERE station in ('{station}') AND programme in ('{programme}');".format(
                station=self.name,
                programme=programme
            )
            dfcontrib = pd.read_sql(sql.format(table="PMF_contributions"),
                                    con=conn)
            dfprofile = pd.read_sql(sql.format(table="PMF_profiles"),
                                    con=conn)
        to_exclude = ["index", "station", "programme"]
        dfcontrib.dropna(axis=1, how="all", inplace=True)
        dfcontrib.drop(to_exclude, axis=1, inplace=True)
        dfcontrib.date = pd.to_datetime(dfcontrib.date)
        dfcontrib.set_index(["date"], drop=True, inplace=True)
        dfprofile.set_index("specie", inplace=True)
        dfcontrib = dfcontrib * dfprofile.loc["PM10", dfcontrib.columns]
        # remove <0 value
        # df[df<0] = 0
        # self.SRC = dfcontrib.sort_index(axis=1).convert_objects(convert_numeric=True)
        self.SRC = dfcontrib.sort_index(axis=1).infer_objects()

    def load_CHEM(self, overwriteBDIR=True):
        df = self.load_from_file(self.CHEMfile, overwriteBDIR)
        # if usecols == None:
        #     df[usecols,
        # TODO

        # warn <0 value
        if (df < 0).any().any():
            print("WARNING: concentration negative in CHEM")
        self.CHEM = df

    def load_OP(self, df=None):
        map_columns_name = {
                "date prelevement": "date",
                "OP_DTT_µg": "DTTm",
                "SD_OP_DTT_µg": "SD_DTTm",
                "OP_DTT_m3": "DTTv",
                "SD_OP_DTT_m3": "SD_DTTv",
                "OP_AA_µg": "AAm",
                "SD_OP_AA_µg": "SD_AAm",
                "OP_AA_m3": "AAv",
                "SD_OP_AA_m3": "SD_AAv",
                "OP_DCFH_µg": "DCFHm",
                "SD_OP_DCFH_µg": "SD_DCFHm",
                "OP_DCFH_m3": "DCFHv",
                "SD_OP_DCFH_m3": "SD_DCFHv"
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
        if (df < 0).any().any():
            print("WARNING: concentration negative in OP")
        if any(df == 0):
            print("WARNING: some value are 0 in OP or SD OP, set it to 1e-5")
            df[df == 0] = 1e-5
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
        possible_sources = {
            "Vehicular": "Traffic",
            "VEH": "Traffic",
            "VEH ind": "Traffic_ind",
            "Traffic_exhaust": "Traffic_exhaust",
            "Traffic_non-exhaust": "Traffic_non-exhaust",
            "VEH dir": "Traffic_dir",
            "Oil/Vehicular": "Traffic",
            "Road traffic": "Traffic",
            "Road trafic": "Traffic",
            "Bio. burning": "Biomass_burning",
            "Bio burning": "Biomass_burning",
            "Comb fossile/biomasse": "Biomass_burning",
            "BB": "Biomass_burning",
            "Biomass Burning": "Biomass_burning",
            "Biomass burning": "Biomass_burning",
            "BB1": "Biomass_burning1",
            "BB2": "Biomass_burning2",
            "Sulfate-rich": "Sulfate_rich",
            "Nitrate-rich": "Nitrate_rich",
            "Sulfate rich": "Sulfate_rich",
            "Nitrate rich": "Nitrate_rich",
            "Secondaire": "Secondary_biogenic",
            "Secondary bio": "Secondary_biogenic",
            "Secondary biogenic": "Secondary_biogenic",
            "Secondary organic": "Secondary_biogenic",
            "Secondaire organique": "Secondary_biogenic",
            "Marine biogenic/HFO": "Marine/HFO",
            "Secondary biogenic/HFO": "Marine/HFO",
            "Marine bio/HFO": "Marine/HFO",
            "Marin bio/HFO": "Marine/HFO",
            "Sulfate rich/HFO": "Marine/HFO",
            "Marine secondary": "Secondary_biogenic",
            "Marin secondaire": "Secondary_biogenic",
            "HFO": "HFO",
            "Marin": "Secondary_biogenic",
            "Sea/road salt": "Salt",
            "Road salt": "Salt",
            "Sea salt": "Salt",
            "Seasalt": "Salt",
            "Fresh seasalt": "Salt",
            "Sels de mer": "Salt",
            "Aged sea salt": "Aged_salt",
            "Aged seasalt": "Aged_salt",
            "Aged seasalt": "Aged_salt",
            "Primary bio": "Primary_biogenic",
            "Primary biogenic": "Primary_biogenic",
            "Biogénique primaire": "Primary_biogenic",
            "Biogenique": "Primary_biogenic",
            "Biogenic": "Primary_biogenic",
            "Mineral dust": "Dust",
            "Mineral dust ": "Dust",
            "Resuspended dust": "Dust",
            "Dust": "Dust",
            "Crustal dust": "Dust",
            "Dust (mineral)": "Dust",
            "Dust/biogénique marin": "Dust",
            "AOS/dust": "Dust",
            "Industrial": "Industrial",
            "Industry": "Industrial",
            "Industrie": "Industrial",
            "Industries": "Industrial",
            "Industry/vehicular": "Indus/veh",
            "Industries/trafic": "Indus/veh",
            "Fioul lourd": "HFO",
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
        Merge the different Biomass burning and Vehicular sources into a
        single one.

        Parameters:
        -----------

        self.SRC: pandas DataFrame
        inplace: boolean, default True. Change the sources in self.SRC.

        Output:
        -------

        df: pandas DataFrame with sources merged.

        """

        df_merged = self.SRC.copy()

        to_merge = [
                ["Bio_burning", "Bio_burning1", "Bio_burning2"],
                ["Vehicular", "Vehicular_ind", "Vehicular_dir"]
                ]

        for source in to_merge:
            if (source[1] in self.SRC.columns or source[2] in self.SRC.columns):
                df_merged[source[0]] = df_merged[source[1:]].sum(axis=1)
                df_merged.drop(source[1:], axis=1, inplace=True)

        if inplace:
            self.SRC = df_merged
        else:
            return df_merged

    def get_WLS_result(self, OPtype, x=None, y=None, sy=None):
        if x is None:
            x = self.OP[OPtype]
            y = self.reg[OPtype].fittedvalues
            idx = x.index.intersection(y.index)

            mywls = sm.WLS(
                        y[idx],
                        sm.add_constant(x[idx]),
                        weights=1/self.OPmodel_unc.loc[idx, OPtype]
                    )
        else:
            mywls = sm.WLS(y, x, sy)
        output = mywls.fit()
        self.wls = mywls
        self.wls_coeff = output.params
        return output

    def get_ODR_result(self, OPtype, x=None, y=None, sx=None, sy=None):
        if x is None:
            idx = self.OP[OPtype].index.intersection(self.OPmodel[OPtype].index)
            data = odr.RealData(
                            self.OP.loc[idx, OPtype],
                            self.OPmodel.loc[idx, OPtype],
                            sx=self.OP.loc[idx, "SD_"+OPtype],
                            sy=self.OPmodel_unc.loc[idx, OPtype]
                        )
        else:
            data = odr.RealData(x, y, sx, sy)
        myodr = odr.ODR(data, odr.unilinear)
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


class Outliers:
    """
    Store the known outliers of station time series.
    """
    def __init__(self, station):

        knownOutliers = {
            "GRE-fr": ["2013-04-26"]
        }

        self.station = station
        self.outliers = {}

        if station in knownOutliers.keys():
            for value in knownOutliers[station]:
                self._addOutlier(k, value)

    def _addOutlier(self, station, date):
        self.outliers[station].append(pd.to_datetime(date))

    def detectOutliers(self):

