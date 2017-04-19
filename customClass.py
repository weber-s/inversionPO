import pandas as pd
import numpy as np
from scipy import polyfit

class Station:
    """
    This class is more a storage class for each station than a proper class...
    """
    def __init__(self,name=None,CHEM=None,PO=None, POunc=None, m=None,covm=None,hasPO=True):
        self.name   = name
        self.CHEM   = CHEM
        self.pieCHEM= CHEM.sum()
        if hasPO:
            self.PO     = PO
            self.POunc  = POunc
            self.m      = m
            self.model  = pd.Series((CHEM*m).sum(axis=1), name=name)
            if (m>=0).all():
                self.pie    = pd.Series((CHEM*m).sum(), name=name)
            else:
                self.pie=pd.Series(index=CHEM.columns)
            self.p      = polyfit(PO,self.model,1)
            self.p.shape= (2,)
            self.r2     = pd.concat([PO,self.model],axis=1).corr().as_matrix()
            self.covm   = covm
        else:
            self.PO     = None
            self.POunc  = None
            self.m      = pd.Series(index=CHEM.columns)
            self.model  = pd.Series(index=CHEM.index)
            self.pie    = pd.Series(index=CHEM.columns)
            self.p      = None
            self.r2     = None
            self.covm   = pd.Series(index=CHEM.columns)
        self.hasPO  = hasPO
        self.covm.sort_index(inplace=True)
        self.pie.sort_index(inplace=True)
        self.m.sort_index(inplace=True)
        self.pieCHEM.sort_index(inplace=True)


