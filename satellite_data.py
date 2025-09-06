import os
import time
import requests
from skyfield.api import load
import math
from datetime import datetime
import pygame

# ==============================================================================
# SATELLITE DATA MANAGEMENT
# ==============================================================================

def download_tle_data(cache_file, tle_url, update_status_callback=None):
    """
    Download TLE data from Celestrak
    Returns the TLE text content
    """
    try:
        if update_status_callback:
            update_status_callback("Downloading TLEs from Celestrak...")
        response = requests.get(tle_url)
        response.raise_for_status()
        tle_text = response.text
        with open(cache_file, 'w') as f:
            f.write(tle_text)
        return tle_text
    except Exception as e:
        if update_status_callback:
            update_status_callback(f"Failed to download TLEs: {str(e)}")
        raise

def load_tle_from_cache(cache_file, update_status_callback=None):
    """
    Load TLE data from cache file
    Returns the TLE text content
    """
    try:
        if update_status_callback:
            update_status_callback("Loading TLEs from cache...")
        with open(cache_file, 'r') as f:
            tle_text = f.read()
        return tle_text
    except Exception as e:
        if update_status_callback:
            update_status_callback(f"Failed to load TLE cache: {str(e)}")
        raise

def is_cache_expired(cache_file, cache_age_limit_seconds):
    """
    Check if cache file is expired
    Returns True if expired or doesn't exist
    """
    if not os.path.exists(cache_file):
        return True

    cache_time = os.path.getmtime(cache_file)
    current_time = time.time()
    return (current_time - cache_time) > cache_age_limit_seconds

def load_satellite_data(cache_file=None, tle_url=None, cache_age_limit_seconds=None, update_status_callback=None):
    """
    Load satellite data from cache or download if needed
    Returns loaded satellite objects and success flag
    """
    if cache_file is None:
        cache_file = "tle_cache.tle"
    if tle_url is None:
        tle_url = 'https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle'
    if cache_age_limit_seconds is None:
        cache_age_limit_seconds = 24 * 3600  # 24 hours

    try:
        if is_cache_expired(cache_file, cache_age_limit_seconds):
            tle_text = download_tle_data(cache_file, tle_url, update_status_callback)
        else:
            tle_text = load_tle_from_cache(cache_file, update_status_callback)

        if update_status_callback:
            update_status_callback("Creating satellite objects...")

        satellites = load.tle_file(cache_file)
        if update_status_callback:
            update_status_callback(f"TLEs ready ({len(satellites)} satellites)")
        return satellites, True

    except Exception as e:
        if update_status_callback:
            update_status_callback(f"Error loading TLEs: {str(e)}")
        return [], False

def create_satellite_labels_and_metadata(satellites, ts):
    """
    Create label renderings and compute orbital parameters for all satellites
    Returns dictionaries for labels, altitudes, perigee, apogee
    """
    MU = 3.986004418e14  # Earth's gravitational parameter in m^3/s^2
    R_EARTH = 6371  # Earth radius in km

    satellite_labels = {}
    satellite_mean_altitudes = {}
    satellite_perigee = {}
    satellite_apogee = {}

    for sat in satellites:
        name = sat.name.strip()
        norad_id = sat.model.satnum_str
        label_text = f"{norad_id} - {name}"
        # Create pygame text surface for the label
        font = pygame.font.Font(None, 16)  # Small font for satellite labels
        satellite_labels[sat] = font.render(label_text, True, (255, 255, 255))  # White text

        # Compute orbital parameters
        n = sat.model.no_kozai / 60  # Mean motion in rad/s
        a = (MU / (n**2))**(1/3) / 1000  # Semi-major axis in km
        e = sat.model.ecco  # Eccentricity

        perigee = a * (1 - e) - R_EARTH  # Perigee altitude in km
        apogee = a * (1 + e) - R_EARTH   # Apogee altitude in km
        mean_altitude = (perigee + apogee) / 2

        satellite_mean_altitudes[sat] = mean_altitude
        satellite_perigee[sat] = perigee
        satellite_apogee[sat] = apogee

    return satellite_labels, satellite_mean_altitudes, satellite_perigee, satellite_apogee
