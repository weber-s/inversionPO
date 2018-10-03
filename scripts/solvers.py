import pandas as pd
import numpy as np
# import pulp
from sklearn import linear_model
import statsmodels.api as sm 
from statsmodels.tools.tools import add_constant

def solve_lsqr(G=None, d=None, Covd=None):
    # G   = G.as_matrix()
    # d   = d.as_matrix()
    C   = np.diag(np.power(Covd,2))

    Gt          = G.T
    invC        = np.linalg.inv(Covd)
    GtinvC      = np.dot(Gt,invC)
    invGtinvCG  = np.linalg.inv(np.dot(GtinvC,G))
    invGtinvCGGt= np.dot(invGtinvCG,Gt)
    Gg          = np.dot(invGtinvCGGt,invC)
    #GtGinvGt=np.dot(linalg.inv(GtG),G.T)
    #r=np.dot(GtGinvGt,b)
    Covm    = np.dot(Gg.dot(Covd),Gg.T)
    Res     = Gg.dot(G)
    m       = Gg.dot(d)

    return m, Covm, Res

def solve_inversion_LP(d, G, std, x_min=None, x_max=None):
    x   = pulp.LpVariable.dicts("OP", G.columns, x_min, x_max)
    m   = pulp.LpVariable("to_minimize", 0)
    lp_prob = pulp.LpProblem("Minmax Problem", pulp.LpMinimize)
    lp_prob += m, "Minimize_the_maximum"

    for i in range(len(G.index)):
        label = "Val %d" % i
        label2 = "Val2 %d" % i
        dot_G_x = pulp.lpSum([G.ix[i][j] * x[j] for j in G.columns])
        condition = (d[i] - dot_G_x) <= m + 0.5*std[i]
        lp_prob += condition, label
        condition = (d[i] - dot_G_x) >= -m - 0.5*std[i]
        lp_prob += condition, label2
    lp_prob.solve()

    return lp_prob

def solve_scikit_linear_regression(X=None, y=None, index=None):
    """
    Solve m*X = y using the sklearn module.
    Different solvers are available.
    """
    if X is None or y is None:
        print("Missing X or y data. Skipping.")
        return
    #regr = linear_model.LinearRegression()
    #regr = linear_model.Lasso(positive=True)
    #regr = linear_model.ElasticNet(l1_ratio=0, positive=True)
    #regr = linear_model.Ridge(alpha=0.01)
    regr = linear_model.Lars(positive=True)
    regr.fit(X, y)
    
    if index is not None:
        m   = dict()
        for i, v in enumerate(regr.coef_):
            m[index[i]] = v
            # print(index[i], "=", v)
        m = pd.Series(m)
    else:
        m = regr.coef_
    return m

def solve_GLS(X=None, y=None, sigma=None):
    """
    Solve a multiple linear problem using statsmodels GLS
    """
    goForGLS = X.copy()
    regr = sm.GLS(y, goForGLS, sigma=sigma).fit()
    while True:
        regr = sm.GLS(y, goForGLS, sigma=sigma).fit()
        # if (regr.pvalues > 0.05).any():
        if (regr.params < 0).any():
            # Some variable are 0, drop them.
            # goForGLS.drop(goForGLS.columns[regr.pvalues>0.05],axis=1,inplace=True)
            # goForGLS.drop(goForGLS.columns[regr.pvalues == max(regr.pvalues)],axis=1,inplace=True)
            goForGLS.drop(goForGLS.columns[regr.params == min(regr.params)],axis=1,inplace=True)
        else:
            # Ok, the run converged
            break
        if goForGLS.shape[1]==0:
            # All variable were droped... Pb
            print("Warning: The run did not converge...")
            break
    # print(regr.summary())
    return regr

def solve_WLS(X=None, y=None, sigma=None):
    """
    Solve a multiple linear problem using statsmodels WLS
    """
    goForWLS = add_constant(X.copy())
    regr = sm.WLS(y, goForWLS, weights=sigma, cov_type="fixed_scale").fit()
    while True:
        regr = sm.WLS(y, goForWLS, weights=sigma, cov_type="fixed_scale").fit()
        # if (regr.pvalues > 0.05).any():
        paramstmp=regr.params.copy()
        paramstmp["const"]=10
        if (paramstmp < 0).any():
            # Some variable are 0, drop them.
            # goForWLS.drop(goForWLS.columns[regr.pvalues>0.05],axis=1,inplace=True)
            # goForWLS.drop(goForWLS.columns[regr.pvalues == max(regr.pvalues)],axis=1,inplace=True)
            print(regr.summary())
            goForWLS.drop(goForWLS.columns[paramstmp == min(paramstmp)],axis=1,inplace=True)
        else:
            # Ok, the run converged
            break
        if goForWLS.shape[1]==0:
            # All variable were droped... Pb
            print("Warning: The run did not converge...")
            break
    # print(regr.summary())
    return regr

def solve_OLS(X=None, y=None, sigma=None):
    """
    Solve a multiple linear problem using statsmodels WLS
    """
    goForWLS = X.copy()
    regr = sm.WLS(y, goForWLS, weights=sigma, cov_type="fixed_scale").fit()
    while True:
        regr = sm.WLS(y, goForWLS, weights=sigma, cov_type="fixed_scale").fit()
        # if (regr.pvalues > 0.05).any():
        if (regr.params < 0).any():
            # Some variable are 0, drop them.
            # goForWLS.drop(goForWLS.columns[regr.pvalues>0.05],axis=1,inplace=True)
            # goForWLS.drop(goForWLS.columns[regr.pvalues == max(regr.pvalues)],axis=1,inplace=True)
            goForWLS.drop(goForWLS.columns[regr.params == min(regr.params)],axis=1,inplace=True)
        else:
            # Ok, the run converged
            break
        if goForWLS.shape[1]==0:
            # All variable were droped... Pb
            print("Warning: The run did not converge...")
            break
    # print(regr.summary())
    return regr


