# hat-creek-skytracker
An autotracker for select telescopes.

# Libraries Utilized

A list of libraries utlilized, to give some sense of the foundation in use.

- SkyField
- Astropy
- Pandas
- Numpy, Scipy
- OpenCV
- Tetra

# Hardaware Interfaces Supported

- Meade LX200 classic interface
- Celestron NexStar AUX interface

# Goals

- Don't replace or re-create everything that SkyTrack can do. Avoid duplication, and focus on creating new capabilities that go beyond it in some niche areas.
- Provide a high quality toolset for finding, acquiring, tracking, and collecting on space objects from stationary or mobile platforms
- Provide a rich, snappy, video game-like UI with modes and prototypes to drive that operation
    - I became tired of and limited by alt-tabbing between too many GUI programs that lack automation-API features to go deeper
    - I need one set of cooperating tools, making optimal use of screen real-estate in good coordination, that lets me set up for and run a selected opportunity, provide insight into what can be achieved by the optical system for this opportunity for the selected object, and then actually run the pass, collect data for the opportunity, and properly label it with all relevant information for later use and post-processing.
    - The experience of tracking satellites should become fun, like a video game, such that even a child could be taught to run it and enjoy the experience.
- JSON Config File and Config Options UI
    - Fixed site location
    - Desired screen layout
        - Enumeration of plots created
        - Plot sizes
        - Plot UL corner locations
    - Enumeration of available hardware interfaces and relation to driver classes
    - Selection of hardware interface
    - Definition of sensor parameters
        - Array size
        - Pixel size
    - Pointer to sensor orientation config file
- UI #1: Visibility and Pass Setup 
    - Detailed fact-extraction, from a few different perspectives.
    - A lot of great software already exists for finding visibilities. We'll let them continue to do that job.
        - Heavens-above.com
        - SkyTrack
        - KStars
        - Various apps
    - Here, we focus on extracting relevant facts for a known opportunity, and providing better sitaware and summary than existing tools
    - Produce a high quality annotated live sky plot that is richer than others available
    - Accept live annotation interactions in the plot for the operator to use to take notes
    - Generation of an angular-ephemeris file for the pass, for use in interpolated time-correlation to imagery
    - Acceptance of a PVT ephemeris file and optional covariance to augment TLE ephemeris
    - Plot and annotate the difference between TLE and Ephemeris-based trajectories on a sky plot
    - Plot and annotate the optical system capability on the sky plot
    - Update this singular plot in real-time as the opportunity progresses
- UI #2: Sensor-to-mount orientation calibration
    - Provide a CLI that runs a two-sensor calibration
    - Provide a simple UI to assist with co-boresighting a guide scope to the main scope against a guide star
    - Provide a simple two-axis calibration against the star to determine sensor rotations with respect to RA/DEC axes, with guidance to co-align both sensors if desired.
    - Output sensor rotations, angular step-change calibrations, and sensor details into main and guide config files.
- UI #3: Joystick Loop with Manual Track, Program Track, Moving Object Search, Acquisition, Tracking, and Auto-labeled Capture
    - Provide a manual joystick loop with rich controls that branches to sub-modes
    - Provide simple programmed following of a TLE trajectory via available hardware interfaces
    - Provide, alternately, a programmed following of an ephemeris trajectory via available hardware interfaces
    - Provide object search while following a programmed trajectory
    - Provide object search while staring, alternatively
    - Provide transition to tracking of a detected object
    - Provide a simple tracked-object tracker, with track "files" (we'll use dictiontaries)
    - Provide hand-picked object selection/override via click-in-image
    - Provide labeled data acquisition as a .png series, with accompanying .json file for each frame containing
        - UTC time
        - Mount az / el
        - Mount rate
        - Detected object centroid
    - Provide capture controls, and joystick integration that allows us to control and direct image/metadata capture easily during a pass, without flipping to other UIs, or wasting too much disk space, which gets painful to manage.
- UI #4: Post Processing UI that provides quick-look and post-processing features
    - It is always a race to post a good shot after an event. You can have the best shot in the world, but if you're a day late, it won't be seen.
    - There are too many tools required to produce a useful product currently
    - In this tool, we integrate a few of those key functions, and speed them up for ease of use