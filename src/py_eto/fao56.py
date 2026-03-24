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


def pm_hourly(t_hr, u2_hr, rn_hr, es_hr, ea_hr, delta_hr, gamma_hr, g_hr=None, reference='short'):
    """
    
    """
    # 1. Determining day or night
    is_day = rn_hr > 0
    
    # 2. Set up Cn values
    cn = 37 if reference == 'short' else 66
    
    # 3. Set up Cd values (based on day or night conditions)
    if reference == 'short':
        cd = np.where(is_day, 0.24, 0.96)
        # 4. Deal with G (if it is None, it will be estimated here)
        if g_hr is None:
            g_hr = np.where(is_day, 0.1 * rn_hr, 0.5 * rn_hr)
    else: # tall reference
        cd = np.where(is_day, 0.25, 1.70)
        if g_hr is None:
            g_hr = np.where(is_day, 0.04 * rn_hr, 0.2 * rn_hr)
            
    # 5. Put them into the core calculation
    return _asce_pm_core(t_hr, u2_hr, rn_hr, g_hr, es_hr, ea_hr, delta_hr, gamma_hr, cn, cd)