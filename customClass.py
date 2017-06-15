import pandas as pd
import numpy as np
from scipy import polyfit

class Station:
    """
    This class is more a storage class for each station than a proper class...
    """
    def __init__(self,name=None,CHEM=None,OP=None, OPunc=None,
                 m=None,covm=None, reg=None, hasOP=True,yerr=None):
        self.name   = name
        self.CHEM   = CHEM
        self.pieCHEM= CHEM.sum()
        if hasOP:
            self.OP     = OP
            self.OPunc  = OPunc
            self.m      = m
            self.model  = pd.Series((CHEM*m).sum(axis=1), name=name)
            if (m>=0).all():
                self.pie    = pd.Series((CHEM*m).sum(), name=name)
            else:
                self.pie=pd.Series(index=CHEM.columns)
            self.p      = polyfit(OP,self.model,1)
            self.p.shape= (2,)
            self.r2     = pd.concat([OP,self.model],axis=1).corr().as_matrix()
            self.covm   = covm
            self.yerr   = yerr 
            self.reg    = reg
        else:
            self.OP     = None
            self.OPunc  = None
            self.m      = pd.Series(index=CHEM.columns)
            self.model  = pd.Series(index=CHEM.index)
            self.pie    = pd.Series(index=CHEM.columns)
            self.p      = None
            self.r2     = None
            self.covm   = pd.Series(index=CHEM.columns)
            self.yerr   = None
            self.reg    = None

        self.hasOP  = hasOP
        self.covm.sort_index(inplace=True)
        self.pie.sort_index(inplace=True)
        self.m.sort_index(inplace=True)
        self.pieCHEM.sort_index(inplace=True)


