from skyfield.api import load, N, W, wgs84
import logging
import sys
from logging.handlers import RotatingFileHandler
logging.basicConfig(
        handlers=[RotatingFileHandler('./vis.log', backupCount=5)],
        level=logging.DEBUG,
        format="[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
        datefmt='%Y-%m-%dT%H:%M:%S')
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

# Create a timescale and ask the current time.
ts = load.timescale()
t = ts.now()

# Load the JPL ephemeris DE421 (covers 1900-2050).
planets = load('de421.bsp')
earth, mars = planets['earth'], planets['mars']

# What's the position of Mars, viewed from Earth?
astrometric = earth.at(t).observe(mars)
ra, dec, distance = astrometric.radec()

# Where am I?
california = earth + wgs84.latlon((34+52/60.0 + 31.8/3600.0) * N, (120 + 26/60.0 + 46.8/3600) * W)

logging.info(ra)
logging.info(dec)
logging.info(distance)

while True:
    t = ts.now()

    astrometric = california.at(t).observe(mars)
    alt, az, d = astrometric.apparent().altaz()

    logging.info(t.utc_iso())
    logging.info(alt)
    logging.info(az)