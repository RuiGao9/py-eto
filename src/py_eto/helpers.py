import numpy as np


def calc_ra(latitude, doy, year):
    """
    Calculate extraterrestrial radiation (Ra)
    Supports scalar values, Numpy arrays, or Pandas Series.
    Torres, A. F., Walker, W. R., & McKee, M. (2011). 
    Forecasting daily potential evapotranspiration using machine learning and limited climatic data. 
    Agricultural Water Management, 98(4), 553-562.
    """
    # Convert latitude to radians
    lat_rad = np.radians(latitude)
    
    # Determine leap year and calculate days in the year
    is_leap = (year % 4 == 0) & ((year % 100 != 0) | (year % 400 == 0))
    days_in_year = np.where(is_leap, 366, 365)
    
    # 1. Declination of the sun
    ds = 0.409 * np.sin((2 * np.pi * doy / days_in_year) - 1.39)
    
    # 2. Relative distance earth-sun
    dr = 1 + 0.033 * np.cos(2 * np.pi * doy / days_in_year)
    
    # 3. Sunset hour angle
    # Correction: Use arccos directly and ensure the input is within the valid range
    tmp = -np.tan(lat_rad) * np.tan(ds)
    # Restrict the range to [-1, 1] to prevent numerical overflow resulting in nan
    tmp = np.clip(tmp, -1, 1)
    ws = np.arccos(tmp)
    
    # 4. Calculate Ra [MJ/(m^2 day)]
    # Constant 37.6 corresponds to Gsc = 0.0820 MJ/m2/min
    ra = (24 * 60 / np.pi) * 0.0820 * dr * (
        ws * np.sin(lat_rad) * np.sin(ds) + 
        np.cos(lat_rad) * np.cos(ds) * np.sin(ws)
    )
    return ra


def calc_delta(t_mean):
    """Calculate slope of saturation vapor pressure curve (delta), kPa/°C"""
    es = 0.6108 * np.exp((17.27 * t_mean) / (t_mean + 237.3))
    delta = (4098 * es) / ((t_mean + 237.3) ** 2)
    return delta


def calc_pressure(elevation):
    """Calculate atmospheric pressure (P), kPa"""
    return 101.3 * ((293 - 0.0065 * elevation) / 293) ** 5.26


def calc_es_ea_no_mean(t_max, t_min, rh_avg=None, rh_max=None, rh_min=None):
    """Calculate saturation vapor pressure (es) and actual vapor pressure (ea)"""
    es_tmax = 0.6108 * np.exp((17.27 * t_max) / (t_max + 237.3))
    es_tmin = 0.6108 * np.exp((17.27 * t_min) / (t_min + 237.3))
    es = (es_tmax + es_tmin) / 2
    if rh_avg is not None:
        ea = (rh_avg / 100) * es
    elif rh_max is not None and rh_min is not None:
        ea = (es_tmin * (rh_max / 100) + es_tmax * (rh_min / 100)) / 2
    else:
        raise ValueError("Either rh_avg or both rh_max and rh_min must be provided.")
    return es, ea


def calc_es_ea(t_mean, rh):
    """Calculate saturation vapor pressure (es) and actual vapor pressure (ea) using mean temperature and relative humidity"""
    es = 0.6108 * np.exp((17.27 * t_mean) / (t_mean + 237.3))
    ea = (rh / 100) * es
    return es, ea


def calc_gamma(pressure, t_mean=None):
    """Calculate psychrometric constant (gamma), kPa/°C"""
    cp = 1.013e-3  # Specific heat of moist air, MJ/kg/°C
    epsilon = 0.622  # Ratio of molecular weight of water vapor/dry air
    if t_mean is not None:
        lambda_v = 2.501 - 0.002361 * t_mean  # Latent heat of vaporization, MJ/kg
    else:
        lambda_v = 2.45  # Default value at 20°C
    tmp = cp / (epsilon * lambda_v)
    return tmp * pressure


def convert_energy(energy):
    """Convert Wm-2 to MJ/m2/day at daily scale"""
    # Convert W/m2 to MJ/m2/day
    return energy * 0.0036