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

def load_satellite_data(state, update_status_callback=None):
    """
    Load satellite data from cache or download if needed
    Modifies state object directly with satellite objects and success flag
    """
    try:
        if is_cache_expired(state.cache_file, state.cache_age_limit):
            download_tle_data(state.cache_file, 'https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle', update_status_callback)
        else:
            load_tle_from_cache(state.cache_file, update_status_callback)

        if update_status_callback:
            update_status_callback("Creating satellite objects...")

        state.satellites = load.tle_file(state.cache_file)
        if update_status_callback:
            update_status_callback(f"TLEs ready ({len(state.satellites)} satellites)")
        state.tle_loaded = True

    except Exception as e:
        if update_status_callback:
            update_status_callback(f"Error loading TLEs: {str(e)}")
        state.satellites = []
        state.tle_loaded = False

def create_satellite_labels_and_metadata(state, update_status_callback=None):
    """
    Create label renderings and compute orbital parameters for all satellites
    Modifies state object directly with labels and orbital parameters
    """
    update_status_callback("Creating satellite labels and metadata...")
    MU = 3.986004418e14  # Earth's gravitational parameter in m^3/s^2
    R_EARTH = 6371  # Earth radius in km

    state.satellite_labels = {}
    state.satellite_mean_altitudes = {}
    state.satellite_perigee = {}
    state.satellite_apogee = {}

    for sat in state.satellites:
        name = sat.name.strip()
        norad_id = sat.model.satnum_str
        label_text = f"{norad_id} - {name}"
        # Create pygame text surface for the label
        font = pygame.font.Font(None, 16)  # Small font for satellite labels
        state.satellite_labels[sat] = font.render(label_text, True, (255, 255, 255))  # White text

        # Compute orbital parameters
        n = sat.model.no_kozai / 60  # Mean motion in rad/s
        a = (MU / (n**2))**(1/3) / 1000  # Semi-major axis in km
        e = sat.model.ecco  # Eccentricity

        perigee = a * (1 - e) - R_EARTH  # Perigee altitude in km
        apogee = a * (1 + e) - R_EARTH   # Apogee altitude in km
        mean_altitude = (perigee + apogee) / 2

        state.satellite_mean_altitudes[sat] = mean_altitude
        state.satellite_perigee[sat] = perigee
        state.satellite_apogee[sat] = apogee
    update_status_callback("Satellite labels and metadata ready.")
