import numpy as np


"""
Equations are following the book:
Task Committee on Revision of Manual 70. 
(2016, April). 
Evaporation, evapotranspiration, and irrigation water requirements. 
American Society of Civil Engineers.
"""
def _asce_pm_core(t_mean, u2, rn, g, es, ea, delta, gamma, cn, cd):
    """ASCE Penman-Monteith algorithm core vectorized calculation logic"""
    numerator = 0.408 * delta * (rn - g) + gamma * (cn / (t_mean + 273)) * u2 * (es - ea)
    denominator = delta + gamma * (1 + cd * u2)
    return numerator / denominator

def pm_daily(t_mean, u2, rn, g, es, ea, delta, gamma, reference='short'):
    """Daily temporal scale for ETo calculation"""
    # Short Reference (Cn=900, Cd=0.34), Tall (Cn=1600, Cd=0.38)
    cn = 900 if reference == 'short' else 1600
    cd = 0.34 if reference == 'short' else 0.38
    return _asce_pm_core(t_mean, u2, rn, g, es, ea, delta, gamma, cn, cd)

def pm_hourly_day(t_mean, u2, rn, g, es, ea, delta, gamma, reference='short'):
    """Hourly temporal scale for ETo calculation (Daytime)"""
    cn = 37 if reference == 'short' else 66
    cd = 0.24 if reference == 'short' else 0.25
    return _asce_pm_core(t_mean, u2, rn, g, es, ea, delta, gamma, cn, cd)

def pm_hourly_night(t_mean, u2, rn, g, es, ea, delta, gamma, reference='short'):
    """Hourly temporal scale for ETo calculation (Nighttime)"""
    cn = 37 if reference == 'short' else 66
    cd = 0.96 if reference == 'short' else 1.7
    return _asce_pm_core(t_mean, u2, rn, g, es, ea, delta, gamma, cn, cd)