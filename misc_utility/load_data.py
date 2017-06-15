import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_CHEMorOP(station, inputDir, CHEMorOP="CHEM", fromSource=True):
    """
    Import the concentration file into a DataFrame

    Parameters
    ----------

    stationName: str, the name of the station

    inputDir: str, the path of the file

    CHEMorOP: str, {"CHEM","OP"}

    fromSource: bool, default True
        with CHEMorOP == "CHEM", chose if the CHEM file is from source or from
        raw chemistry.
    
    Output
    ------

    df: a panda DataFrame
    """
    if CHEMorOP == "CHEM":
        if fromSource:
            nameFile = "_ContributionsMass.csv"
        else:
            nameFile = "CHEM_conc.csv"
    elif CHEMorOP == "OP":
        # nameFile = "OP_with_outliers.csv"
        nameFile = "OP.csv"
    else:
        raise TypeError("CHEMorOP must be 'CHEM' or 'OP'")

    df = pd.read_csv(inputDir+station+"/"+station+nameFile,
                     index_col="date", parse_dates=["date"],
                     na_values=['#VALUE!', '#DIV/0!'])
    if CHEMorOP == "CHEM":
        df[df<0] = 0
    else:
        for i in df.columns:
            if "SD" in i:
                pass
            else:
                # df["SD_"+i][df[i]<0] = df["SD_"+i].min()*8
                df["SD_"+i][df[i]<0] = df["SD_"+i][df[i]>0].max()
                df[i][df[i]<0] = df[i][df[i]>0].min()
    df.name = station
    return df

def setSourcesCategories(df):
    """
    Return the DataFrame df with a renamed columns.
    The renaming axes set the source's name to its category, i.e
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
    
    return df.rename(columns=possible_sources)
