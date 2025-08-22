from collections import defaultdict 

class CameraConfig:
    """Container for non-manufacturer specific camera configurations"""
    def __init__(self):
        self.xpixels = defaultdict(self.def_value)
        self.ypixels = defaultdict(self.def_value)
        self.xsize = defaultdict(self.def_value)
        self.ysize = defaultdict(self.def_value)
        self.adc = defaultdict(self.def_value)
        self.pixel_size = defaultdict(self.def_value)
        self.fps = defaultdict(self.def_value)
        self.fullwell = defaultdict(self.def_value)
        
        self.define_camera("ASI178MC", 3096, 2080, 7.4, 5.0, 14, 2.4, 60, 15000)
        self.define_camera("ASI462MM", 1936, 1096, 5.6, 3.2, 12, 2.9, 136, 11200)
    
    def define_camera(self, key, xpixels, ypixels, xsize, ysize, adc, pixel_size, fps, fullwell):
        """Define a generic camera
        
        Args:
            key (str): Name for the camera, also used as the dict key
            xpixels (int): X size of the camera pixel array
            ypixels (int): Y size of the camera pixel array
            xsize (float): X size of the focal plane array in millimeters
            ysize (float): Y size of the focal plane array in millimeters
            adc (int): ADC depth of the sensor readout (bits)
            pixel_size (float): Pixel size in micrometers
            fps (int): Frame rate in frames per second
            fullwell (int): Fullwell capacity in electron count
        """
        self.xpixels[key] = xpixels
        self.ypixels[key] = ypixels
        self.xsize[key] = xsize
        self.ysize[key] = ysize
        self.adc[key] = adc
        self.pixel_size[key] = pixel_size
        self.fps[key] = fps
        self.fullwell[key] = fullwell
        
    def def_value(self): 
        return "Not Present"
    
    def cam_names(self):
        return self.xpixels.keys()