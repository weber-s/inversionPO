import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_CHEMorPO(station, inputDir, CHEMorPO="CHEM", fromSource=True):
    """
    Import the concentration file into a DataFrame

    Parameters
    ----------

    stationName: str, the name of the station

    inputDir: str, the path of file

    CHEMorPO: str, {"CHEM","PO"}

    fromSource: bool, default True
        whith CHEMorPO == "CHEM", chose if the CHEM file is from source or from
        raw chemistry.
    
    Output
    ------

    df: a panda DataFrame
    """
    if CHEMorPO == "CHEM":
        if fromSource:
            nameFile = "_ContributionsMass_positive.csv"
        else:
            nameFile = "CHEM_conc.csv"
    elif CHEMorPO == "PO":
        nameFile = "PO.csv"
    else:
        print("Error: CHEMorPO must be 'CHEM' or 'PO'")
        return 

    df = pd.read_csv(inputDir+station+"/"+station+nameFile,
                     index_col="date", parse_dates=["date"])
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
        "Bio. burning": "Bio_burning",
        "Bio burning": "Bio_burning",
        "BB": "Bio_burning",
        "BB1": "Bio_burning1",
        "BB2": "Bio_burning2",
        "Sulfate-rich": "Sulfate_rich",
        "Nitrate-rich": "Nitrate_rich",
        "Sulfate rich": "Sulfate_rich",
        "Nitrate rich": "Nitrate_rich",
        "Secondaire": "Secondary_bio",
        "Secondary bio": "Secondary_bio",
        "Secondary biogenic": "Secondary_bio",
        "Marine biogenic/HFO": "Marine_bio/HFO",
        "Marine bio/HFO": "Marine_bio/HFO",
        "Marin bio/HFO": "Marine_bio/HFO",
        "Marine secondary": "Marine_bio",
        "Marin secondaire": "Marine_bio",
        "HFO": "HFO",
        "Marin": "Marine",
        "Sea/road salt": "Salt",
        "Sea salt": "Salt",
        "Aged sea salt": "Aged_salt",
        "Aged seasalt": "Aged_salt",
        "Primary bio": "Primary_bio",
        "Primary biogenic": "Primary_bio",
        "Biogenique": "Primary_bio",
        "Biogenic": "Primary_bio",
        "Mineral dust": "Dust",
        "Resuspended dust": "Dust",
        "Dust": "Dust",
        "Dust (mineral)": "Dust",
        "AOS/dust": "Dust",
        "Industrial": "Industrial",
        "Industry/vehicular": "Indus._veh.",
        "Arcellor": "Industrial",
        "Siderurgie": "Industrial",
        "Débris végétaux": "Plant_debris",
        "Chlorure": "Chloride",
        "PM other": "Other"
    }
    
    return df.rename(columns=possible_sources)
