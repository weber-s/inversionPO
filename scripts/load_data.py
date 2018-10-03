import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def getDataFromDB(conn, programmes=None):
    sqlqueryPMF = "SELECT * FROM {table} WHERE programme IN ('{p}');"
    sqlqueryChemistry = "SELECT * FROM {table};"

    contributions = pd.read_sql(
        sqlqueryPMF.format(
            table="PMF_contributions",
            p="', '".join(programmes)
        ),
        con=conn
    )
    profiles = pd.read_sql(
        sqlqueryPMF.format(
            table="PMF_profiles",
            p="', '".join(programmes)
        ),
        con=conn
    )
    chemistry = pd.read_sql(
        sqlqueryChemistry.format(
            table="values_all"
        ),
        con=conn
    )

    # drop NaN
    contributions.dropna(axis=1, how="all", inplace=True)

    return (contributions, profiles, chemistry)


def load_CHEMorOP(station, inputDir, CHEMorOP="CHEM", fromSource=True,
                  usecols=None):
    """
    Import the concentration file into a DataFrame

    Parameters
    ----------

    station: str, the name of the station

    inputDir: str, the path of the file

    CHEMorOP: str, "CHEM" or "OP"

    fromSource: bool, default True
        with CHEMorOP == "CHEM", chose if the CHEM file is from source or from
        raw chemistry.

    usecols: default None, or a list if fromSource=False and CHEMorOP="CHEM".
        The columns to use from the CSV file. The default chemical species are:
            usecols = ("OC","EC",\
                       "Na+","NH4+","K+","Mg2+","Ca2+","MSA","Cl-","NO3-","SO42-","Oxalate",\
                       "Levoglucosan","Levoglusan","Polyols","ΣPolyols",\
                       "As","Ba","Cd","Cu","Hg","Mn","Mo","Ni","Pb","Sb","Sn","Ti","Zn","Al","Fe","Ag"\
                       "ΣHAP","ΣHOP","Σmethoxy_part")
    
    Output
    ------

    df: a pandas DataFrame
    """

    if CHEMorOP == "CHEM":
        if fromSource:
            nameFile = "_ContributionsMass_Florie.csv"
        else:
            nameFile = "CHEM_conc.csv"
            # select the species we want
            if usecols is None:
                usecols  = ["OC","EC",\
                            "Na+","NH4+","K+","Mg2+","Ca2+","MSA","Cl-","NO3-","SO42-","Oxalate",\
                            "Levoglucosan","Levoglusan","Polyols","ΣPolyols",\
                            "As","Ba","Cd","Cu","Hg","Mn","Mo","Ni","Pb","Sb","Sn","Ti","Zn","Al","Fe","Ag"\
                            "ΣHAP","ΣHOP","Σmethoxy_part"]
    elif CHEMorOP == "OP":
        # nameFile = "OP_with_outliers.csv"
        nameFile = "OP.csv"
    else:
        raise TypeError("`CHEMorOP` must be 'CHEM' or 'OP'")

    df = pd.read_csv(inputDir+station+"/"+station+nameFile,
                     usecols=usecols,
                     index_col=["date"], parse_dates=["date"],
                     na_values=['#VALUE!', '#DIV/0!'])


    # ensure positivity 
    if CHEMorOP == "CHEM":
        df[df<0] = 0
    else:
        for i in df.columns:
            if "SD" in i or "date" in i:
                pass
            else:
                # df["SD_"+i][df[i]<0] = df["SD_"+i].min()*8
                df["SD_"+i][df[i]<0] = df["SD_"+i][df[i]>0].max()
                df[i][df[i]<0] = df[i][df[i]>0].min()

    # set the name of the DataFrame
    df.name = station
    return df

def load_data(station, SRC=False, OP=False, CHEM=False, usecols=None):
    """
    Import the concentration file into a DataFrame

    Parameters
    ----------

    station: Station object

    SRC: boolean, default False. Load PMF sources contribution.
    
    OP: boolean, default False. Load OP values.

    CHEM: boolean, default False. Load chemistry value.

    Output
    ------

    df: a pandas DataFrame
    """

    basename = station.inputDir+station.name+"/"+station.name
    file = []
    if SRC:
        file.append(basename + "_ContributionsMass_Florie.csv")
    if OP:
        file.append(basename + "OP.csv")
    if CHEM:
        file.append(basename + "CHEM_conc.csv")
        if usecols is None:
            usecols  = ["OC","EC",\
                        "Na+","NH4+","K+","Mg2+","Ca2+","MSA","Cl-","NO3-","SO42-","Oxalate",\
                        "Levoglucosan","Levoglusan","Polyols","ΣPolyols",\
                        "As","Ba","Cd","Cu","Hg","Mn","Mo","Ni","Pb","Sb","Sn","Ti","Zn","Al","Fe","Ag"\
                        "ΣHAP","ΣHOP","Σmethoxy_part", "date"]

    if not (SRC or OP or CHEM):
        raise TypeError("You must provide SRC, OP or CHEM")

    dfout = []
    for f in file:
        df = pd.read_csv(f,
                         usecols=usecols,
                         index_col=["date"], parse_dates=["date"],
                         na_values=['#VALUE!', '#DIV/0!'])

        # ensure positivity 
        if "OP" not in f:
            df[df<0] = 0
        else:
            for i in df.columns:
                if "SD" in i or "date" in i:
                    pass
                else:
                    # df["SD_"+i][df[i]<0] = df["SD_"+i].min()*8
                    df["SD_"+i][df[i]<0] = df["SD_"+i][df[i]>0].max()
                    df[i][df[i]<0] = df[i][df[i]>0].min()
        df.name = station.name
        dfout.append(df)

    return dfout

def load_station(name, list_OPtype, inputDir):
    """
    Load the Station saved in `inputDir/name.pickle`
    """

    data = dict()

    for OPtype in list_OPtype:
        with open(inputDir+"/"+name+OPtype+".pickle","rb") as f:
            data[OPtype] = pickle.load(f)

    return data

def setSourcesCategories(df):
    """
    Return the DataFrame or an array `df` with a renamed columns/item.
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
        "Crustal dust": "Crustal_dust",
        "Dust (mineral)": "Dust",
        "Dust/biogénique marin": "Dust",
        "AOS/dust": "Dust",
        "Industrial": "Industrial",
        "Industrie": "Industrial",
        "Industry/vehicular": "Indus/veh",
        "Industries/trafic": "Vehicular",
        "Fioul lourd": "Industrial",
        "Arcellor": "Industrial",
        "Siderurgie": "Industrial",
        "Débris végétaux": "Plant_debris",
        "Chlorure": "Chloride",
        "PM other": "Other"
    }
    if type(df) is list:
        return [possible_sources[x] for x in df]
    else:
        return df.rename(columns=possible_sources)

def mergeSources(df):
    """
    Merge the different Biomass burning and Vehicular sources into a single one.

    Parameters:
    -----------
    
    df: pandas DataFrame

    Output:
    -------

    df_merged: pandas DataFrame with sources merged.

    """
    
    df_merged= df.copy()

    to_merge =[["Bio_burning", "Bio_burning1","Bio_burning2"],\
               ["Vehicular", "Vehicular_ind","Vehicular_dir"]]

    for source in to_merge:
        if (source[1] in df.columns or source[2] in df.columns): 
            df_merged[source[0]] = df_merged[source[1:]].sum(axis=1)
            df_merged.drop(source[1:], axis=1,inplace=True)

    return df_merged
