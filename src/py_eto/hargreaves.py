import numpy as np
from .helpers import calc_ra

def hargreaves(t_min, t_max, latitude, doy, year=2024):
    """
    Calculate daily reference evapotranspiration using the Hargreaves-Samani (1985) formula.
    Required inputs:
    - t_min: Minimum daily temperature (°C)
    - t_max: Maximum daily temperature (°C)
    - latitude: Latitude of the location (degrees)
    - doy: Day of the year (1-365/366)
    - year: Year (default is 2024)

    """
    # Calculate mean temperature
    t_mean = (t_max + t_min) / 2.0
    
    # Get Ra
    ra = calc_ra(latitude, doy, year)
    
    # Temperature difference (Tmax - Tmin)
    tdiff = np.maximum(t_max - t_min, 0)
    
    # Hargreaves formula: 0.0023 * (Tmean + 17.8) * (Tdiff ^ 0.5) * (0.408 * Ra)
    # 0.408 is used to convert energy units (MJ/m2/d) to depth units (mm/d)
    eto = 0.0023 * (t_mean + 17.8) * np.sqrt(tdiff) * (0.408 * ra)
    
    return eto