# src/merton_solver.py
import numpy as np
from scipy.optimize import root, least_squares
from scipy.stats import norm
import pandas as pd

def merton_residuals(x, E, sigma_e, D, r, T):
    V, sigma_v = x
    if V <= 0 or sigma_v <= 1e-8:
        return [1e9,1e9]
    sqrtT = np.sqrt(T)
    d1 = (np.log(V/D) + (r + 0.5*sigma_v**2)*T) / (sigma_v*sqrtT)
    d2 = d1 - sigma_v*sqrtT
    Nd1, Nd2 = norm.cdf(d1), norm.cdf(d2)
    eq1 = V*Nd1 - D*np.exp(-r*T)*Nd2 - E
    eq2 = sigma_v*V*Nd1 - sigma_e*E
    return [eq1, eq2]

def solve_merton(E, sigma_e, D, r, T=1.0):
    V0 = max(E + D, E*1.2 + D*0.8)
    sigma_v0 = max(0.2, (sigma_e * E) / (V0 if V0>0 else 1))
    x0 = np.array([V0, sigma_v0])
    sol = root(merton_residuals, x0, args=(E, sigma_e, D, r, T), method='hybr', tol=1e-10)
    if not sol.success:
        sol_ls = least_squares(lambda x: np.array(merton_residuals(x, E, sigma_e, D, r, T)), x0, bounds=([1e-6,1e-8],[np.inf,10.0]))
        sol_x = sol_ls.x
    else:
        sol_x = sol.x
    V_est, sigma_v_est = sol_x[0], sol_x[1]
    sqrtT = np.sqrt(T)
    d1 = (np.log(V_est/D) + (r + 0.5*sigma_v_est**2)*T) / (sigma_v_est*sqrtT)
    d2 = d1 - sigma_v_est*sqrtT
    PD = norm.cdf(-d2)
    return {"V":V_est, "Sigma_V":sigma_v_est, "d1":d1, "d2":d2, "PD":PD}