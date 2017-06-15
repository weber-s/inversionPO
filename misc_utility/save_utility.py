import numpy as np
import pandas as pd
import os

def result2csv(station,saveDir=None,OPtype=None):
    """
    Save the station object into a csv file.
    """
    if saveDir == None:
        saveDir = "./"
    
    if not os.path.isdir(saveDir):
        os.makedirs(saveDir)
    
    save_name = saveDir + station.name + "_contribution_"+OPtype+".csv"

    df = station.CHEM * station.m
    df.to_csv(save_name)

