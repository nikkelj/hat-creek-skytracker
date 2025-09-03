
#! /usr/bin/env python3
# ^doesn't do anything for us in windows...we call interpreter explicitly

import ctypes
import h5py
import numpy as np #numpy math library
import time 
import cv2 #opencv library

#typedef struct _ASI_CAMERA_INFO
#{
#    char Name[64]; //the name of the camera, you can display this to the UI
#    int CameraID; //this is used to control everything of the camera in other functions
#    long MaxHeight; //the max height of the camera
#    long MaxWidth;  //the max width of the camera
#
#    ASI_BOOL IsColorCam;
#    ASI_BAYER_PATTERN BayerPattern;
#
#    int SupportedBins[16]; //1 means bin1 which is supported by every camera, 2 means bin 2 etc.. 0 is the end of supported binning method
#    ASI_IMG_TYPE SupportedVideoFormat[8]; //this array will content with the support output format type.IMG_END is the end of supported video format
#
#    double PixelSize; //the pixel size of the camera, unit is um. such like 5.6um
#    ASI_BOOL MechanicalShutter;
#    ASI_BOOL ST4Port;
#    ASI_BOOL IsCoolerCam;
#    ASI_BOOL IsUSB3Host;
#    char Unused[32];
#} ASI_CAMERA_INFO;

class ASI_CAMERA_INFO(ctypes.Structure):
    _fields_ = [
        ("Name"                 , ctypes.c_char * 64),
        ("CameraID"             , ctypes.c_int      ),
        ("MaxHeight"            , ctypes.c_long     ),
        ("MaxWidth"             , ctypes.c_long     ),
        ("IsColorCam"           , ctypes.c_int      ),
        ("BayerPattern"         , ctypes.c_int      ),
        ("SupportedBins"        , ctypes.c_int * 16 ),
        ("SupportedVideoFormat" , ctypes.c_int * 8  ),
        ("PixelSize"            , ctypes.c_double   ),
        ("MechanicalShutter"    , ctypes.c_int      ),
        ("ST4Port"              , ctypes.c_int      ),
        ("IsCoolerCam"          , ctypes.c_int      ),
        ("IsUSB3Host"           , ctypes.c_int      ),
        ("Unused"               , ctypes.c_char * 32)
    ]

class ASI_CONTROL_CAPS(ctypes.Structure):
    _fields_ = [
        ("Name"            , ctypes.c_char * 64 ),
        ("Description"     , ctypes.c_char * 128),
        ("MaxValue"        , ctypes.c_long      ),
        ("MinValue"        , ctypes.c_long      ),
        ("DefaultValue"    , ctypes.c_long      ),
        ("IsAutoSupported" , ctypes.c_int       ),
        ("IsWritable"      , ctypes.c_int       ),
        ("ControlType"     , ctypes.c_int       ),
        ("Unused"          , ctypes.c_char * 32 )
    ]
    
    
_errorcodes = {0: "ASI_SUCCESS",
	1:"ASI_ERROR_INVALID_INDEX", # //no camera connected or index value out of boundary
	2:"ASI_ERROR_INVALID_ID", # //invalid ID
	3:"ASI_ERROR_INVALID_CONTROL_TYPE", # //invalid control type
	4:"ASI_ERROR_CAMERA_CLOSED", # //camera didn't open
	5:"ASI_ERROR_CAMERA_REMOVED", # //failed to find the camera, maybe the camera has been removed
	6:"ASI_ERROR_INVALID_PATH", # //cannot find the path of the file
	7:"ASI_ERROR_INVALID_FILEFORMAT", # 
	8:"ASI_ERROR_INVALID_SIZE",# //wrong video format size
	9:"ASI_ERROR_INVALID_IMGTYPE", #//unsupported image formate
	10:"ASI_ERROR_OUTOF_BOUNDARY",# //the startpos is out of boundary
	11:"ASI_ERROR_TIMEOUT",# //timeout
	12:"ASI_ERROR_INVALID_SEQUENCE",#//stop capture first
	13:"ASI_ERROR_BUFFER_TOO_SMALL",# //buffer size is not big enough
	14:"ASI_ERROR_VIDEO_MODE_ACTIVE",#
	15:"ASI_ERROR_EXPOSURE_IN_PROGRESS",#
	16:"ASI_ERROR_GENERAL_ERROR",#//general error, eg: value is out of valid range
	17:"ASI_ERROR_END"}
    
_exposurecodes = {0:"ASI_EXP_IDLE", #//: idle states, you can start exposure now
	1:"ASI_EXP_WORKING",#//: exposing
	2:"ASI_EXP_SUCCESS",#// exposure finished and waiting for download
	3:"ASI_EXP_FAILED"}

_imgtypes = {0:"ASI_IMG_RAW8",
	1:"ASI_IMG_RGB24",
	2:"ASI_IMG_RAW16",
	3:"ASI_IMG_Y8",
	4:"ASI_IMG_END"}


#libasi = ctypes.cdll.LoadLibrary("../libASICamera2.so.0.1.0320")
libasi = ctypes.cdll["ASICamera2"]

numberOfCameras = libasi.ASIGetNumOfConnectedCameras()
print("Num Connected Cameras:" + str(numberOfCameras))

for cameraIndex in range(numberOfCameras):

    cameraInfo = ASI_CAMERA_INFO()

    #Get Camera Properties
    errorcode = libasi.ASIGetCameraProperty(ctypes.byref(cameraInfo), cameraIndex)
    try:
        assert errorcode == 0
    except:
        print("Get Cam Properties Error Code: " + str(_errorcodes[errorcode]))
    
    #Open Camera
    errorcode = libasi.ASIOpenCamera(cameraInfo.CameraID)
    try:
        assert errorcode == 0
        print("Camera " + str(cameraInfo.CameraID) + " opened.")
    except:
        print("Open Camera Error Code: " + str(_errorcodes[errorcode]))
    
    #Init Camera 
    errorcode = libasi.ASIInitCamera(cameraInfo.CameraID)
    try:
        assert errorcode == 0
        print("Camera " + str(cameraInfo.CameraID) + " inited.")
    except:
        print("Init Camera Error Code: " + str(_errorcodes[errorcode]))

    
    #Get Controls
    numberOfControls = ctypes.c_int()
    errorcode = libasi.ASIGetNumOfControls(cameraInfo.CameraID, ctypes.byref(numberOfControls))
    try:
        assert errorcode == 0
    except:
        print("Get Num Controls Error Code: " + str(_errorcodes[errorcode]))
        
    print("number of controls:", numberOfControls.value)

    controlInfo = ASI_CONTROL_CAPS()

    #Get Control Value Info and Current Settings
    print("Getting Control Parameter Info...")
    for controlIndex in range(numberOfControls.value):
        print("\t"+str(controlIndex)+":")
        errorcode = libasi.ASIGetControlCaps(cameraInfo.CameraID, controlIndex, ctypes.byref(controlInfo))
        try:
            assert errorcode == 0
        except:
            print("Get Control Caps Error Code: " + str(_errorcodes[errorcode]))
        print("\t\tName: "+str(controlInfo.Name))
        print("\t\tDescription: "+str(controlInfo.Description))
        print("\t\tMaxValue: "+str(controlInfo.MaxValue))
        print("\t\tMinValue: "+str(controlInfo.MinValue))
        print("\t\tDefaultValue: "+str(controlInfo.DefaultValue))
        print("\t\tIsAutoSupported: "+str(controlInfo.IsAutoSupported))
        print("\t\tIsWritable: "+str(controlInfo.IsWritable))
        print("\t\tControlType: "+str(controlInfo.ControlType))
        cvalue = ctypes.c_int(-1)
        cauto = ctypes.c_int(-1)
        errorcode = libasi.ASIGetControlValue(cameraInfo.CameraID, controlIndex, ctypes.byref(cvalue), ctypes.byref(cauto))
        try:
            assert errorcode == 0
            print("\t\t"+str(controlInfo.Name) + " Settings:")
        except:
            print("Get Control Value Error Code: " + str(_errorcodes[errorcode]))
        print("\t\t\t" + str(controlInfo.Name) + " Current Value: " + str(cvalue.value))
        print("\t\t\t" + str(controlInfo.Name) + " Current Auto: " + str(cauto.value))

    print("")
    print("Apply Settings:")
    #Apply any Custom Control Value Settings you want here by copying these code blocks...
    
    #Set Exposure Time in microseconds (control index = 1)
    #Min = 64 us
    #Max = 2000000000us (2000 seconds)
    errorcode = libasi.ASISetControlValue(cameraInfo.CameraID, 1, 10000, False);
    try:
        assert errorcode == 0
        print("Exposure Time Value is set.")
    except:
        print("Set Control Value Error Code: " + str(_errorcodes[errorcode]))
    
    #Get Exposure Setting
    expvalue = ctypes.c_int(-1)
    expauto = ctypes.c_int(-1)
    errorcode = libasi.ASIGetControlValue(cameraInfo.CameraID, 1, ctypes.byref(expvalue), ctypes.byref(expauto))
    try:
        assert errorcode == 0
        print("Exposure Settings:")
    except:
        print("Get Control Value Error Code: " + str(_errorcodes[errorcode]))
    print("\tExposure Value: " + str(expvalue.value) + " us")
    print("\tExposure Auto: " + str(expauto.value))
    
    
    #Set Gain
    #Higher Gain = Higher Noise.  Since we can frame stack, we should keep this low, with auto disabled.
    #Min = 0
    #Max = 100
    #Default = 50
    #Has Auto Gain
    errorcode = libasi.ASISetControlValue(cameraInfo.CameraID, 0, 0, False);
    try:
        assert errorcode == 0
        print("Gain Value is set.")
    except:
        print("Set Gain Value Error Code: " + str(_errorcodes[errorcode]))
    
    #Get Exposure Setting
    gainvalue = ctypes.c_int(-1)
    gainauto = ctypes.c_int(-1)
    errorcode = libasi.ASIGetControlValue(cameraInfo.CameraID, 0, ctypes.byref(gainvalue), ctypes.byref(gainauto))
    try:
        assert errorcode == 0
        print("Gain Settings:")
    except:
        print("Get Gain Value Error Code: " + str(_errorcodes[errorcode]))
    print("\tGain Value: " + str(gainvalue.value) + "")
    print("\tGain Auto: " + str(gainauto.value))
    
    
    
    #Set the ROI
    errorcode = libasi.ASISetROIFormat(cameraInfo.CameraID, 1280, 960, 1, 2)
    try:
        assert errorcode == 0
        print("ROI is set.")
    except:
        print("Set ROI Error Code: " + str(_errorcodes[errorcode]))
        
    #Get the ROI Settings
    width = ctypes.c_int(-1)
    height = ctypes.c_int(-1)
    bintype = ctypes.c_int(-1)
    imgtype = ctypes.c_int(-1)
    errcode = libasi.ASIGetROIFormat(cameraInfo.CameraID,ctypes.byref(width),ctypes.byref(height),ctypes.byref(bintype),ctypes.byref(imgtype))
    try:
        assert errorcode == 0
        print("ROI settings:")
    except:
        print("Get ROI Error Code: " + str(_errorcodes[errorcode]))
    print("\tWidth: " + str(width.value))
    print("\tHeight: " + str(height.value))
    print("\tBinType: " + str(bintype.value))
    print("\tImgType: " + _imgtypes[imgtype.value])
    
    
    #Set the Start Position (important if ROI'ing)
    errorcode = libasi.ASISetStartPos(cameraInfo.CameraID,0,0)
    try:
        assert errorcode == 0
        print("Successfully set ROI start position.")
    except:
        print("Set StartPos Error Code: " + str(_errorcodes[errorcode]))
    
    
    #Get the Start Position Settings
    startx = ctypes.c_int(-1)
    starty = ctypes.c_int(-1)
    errorcode = libasi.ASIGetStartPos(cameraInfo.CameraID,ctypes.byref(startx),ctypes.byref(starty))
    try:
        assert errorcode == 0
        print("StartPos settings:")
    except:
        print("Get StartPos Error Code: " + str(_errorcodes[errorcode]))
    print("\tStartX: " + str(startx.value))
    print("\tStartY: " + str(starty.value))
        
    #print("starting capture in 3...")
    #time.sleep(3)

    #Set up capture data structures
    bb = np.zeros((960, 1280), dtype = 'u2')          #base image - we accumulate the "stacked" moving average into this buffer
    fb = np.zeros((960, 1280), dtype = 'u2')    #front image - most recently sampled image
    f = h5py.File("images.h5", "w")
    images = f.create_dataset("images", shape = (0, 960, 1280), maxshape = (None, 960, 1280), dtype = 'u2')

    #Do some captures
    ccount = 0
    while (1):
        print("-----------------------------------------------------")
        print("Frame Count: " + str(ccount))
        print("Getting Exposure Status...")
        expstatus = ctypes.c_int(-1)
        errorcode = libasi.ASIGetExpStatus(cameraInfo.CameraID,ctypes.byref(expstatus))
        print("Get Exposure Status Error Code: " + str(_errorcodes[errorcode]))
        print("Exposure Status Code: " + str(_exposurecodes[expstatus.value]))
        
        print("Starting Exposure...")
        errorcode = libasi.ASIStartExposure(cameraInfo.CameraID)
        try:
            assert errorcode == 0
            print("Exposure Started.")
        except:
            print("Start Exposure Error Code: " + str(_errorcodes[errorcode]))

        print("Getting Exposure Status...")
        expstatus = ctypes.c_int(-1)
        errorcode = libasi.ASIGetExpStatus(cameraInfo.CameraID,ctypes.byref(expstatus))
        print("Get Exposure Status Error Code: " + str(_errorcodes[errorcode]))
        print("Exposure Status Code: " + str(_exposurecodes[expstatus.value]))
        
        #Wait for Success Status Before Trying to Read
        totalwait = 0
        while expstatus.value != 2:
                    
            waitsecs = 0.01
            time.sleep(waitsecs)
            #print("Waiting " + str(waitsecs) + " seconds...")
            
            #print("Getting Exposure Status...")
            expstatus = ctypes.c_int(-1)
            errorcode = libasi.ASIGetExpStatus(cameraInfo.CameraID,ctypes.byref(expstatus))
            #print("Get Exposure Status Error Code: " + str(_errorcodes[errorcode]))
            #print("Exposure Status Code: " + str(_exposurecodes[expstatus.value]))
            totalwait += waitsecs
        
        print("")
        print("Total Wait Time: " + str(totalwait) + " seconds")
        print("")
        
        #If good exposure, read it out
        if expstatus.value == 2:
            #frame = 0
            print("Reading exposure.")
            errorcode = libasi.ASIGetDataAfterExp(cameraInfo.CameraID, fb.ctypes, 960 * 1280 * 2)
            try:
                assert errorcode == 0
            except:
                print("Get Exposure Data Error Code: " + str(_errorcodes[errorcode]))
            #images.resize(frame + 1, axis = 0)
            #images[frame] = (fb + frame)
            #print(images.shape)
        else:
            print("Bad exposure status, skipping read.")

        #f.close()
        
        #Stop Exposure
        print("Stopping Exposure...")
        errorcode = libasi.ASIStopExposure(cameraInfo.CameraID)
        try:
            assert errorcode == 0
            print("Exposure Stopped.")
        except:
            print("Stop Exposure Error Code: " + str(_errorcodes[errorcode]))
            
        print("Getting Exposure Status...")
        expstatus = ctypes.c_int(-1)
        errorcode = libasi.ASIGetExpStatus(cameraInfo.CameraID,ctypes.byref(expstatus))
        print("Get Exposure Status Error Code: " + str(_errorcodes[errorcode]))
        print("Exposure Status Code: " + str(_exposurecodes[expstatus.value]))
        
        #Do the image stacking math
        #Note: we intentionally store a floating point history, not a uint16 history, as we want all that information.
        #When we display, we must truncate of course, but the fractional bin info is important when summing
        if ccount > 0:
            bb = (bb.astype(np.float) + fb.astype(np.float))/8.0  #change 2 to any integer number to make a longer "stack" (average)
        else:
            bb = fb
        
        #Convert and Display Base (Stacked) Image Using OpenCV
        #Once you hit a key, it will do the next capture
        cv2.imshow('dst',(8*bb).astype(np.uint16)) #change to fb if want show current only
        cv2.waitKey(0)
        
        ccount += 1

    #end while(1)
    
    
    #Cleanup Stuff
    #Kill the display window
    cv2.destroyAllWindows()
          
    #Close Camera
    errorcode = libasi.ASICloseCamera(cameraInfo.CameraID)
    try:
        assert errorcode == 0
        print("Camera closed.")
    except:
        print("Close Camera Error Code: " + str(_errorcodes[errorcode]))
        
        
