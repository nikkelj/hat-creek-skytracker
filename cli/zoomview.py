import argparse
import logging
import os
import time
import sys
from logging.handlers import RotatingFileHandler

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# Logging setup
logging.basicConfig(
        handlers=[RotatingFileHandler('./zoomview.log', backupCount=5)],
        level=logging.DEBUG,
        format="[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
        datefmt='%Y-%m-%dT%H:%M:%S')
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
logging.getLogger('matplotlib.font_manager').disabled = True

parser = argparse.ArgumentParser(
                    prog='zoomview.py',
                    description='Zoom-view an image sequence')
parser.add_argument("--directory", type=str, help='Directory to process')
parser.add_argument("--wait", type=int, help="Milliseconds of wait between frames")
args = parser.parse_args()


alpha = 1.0
alpha_max = 500
beta = 0
beta_max = 200
gamma = 0.1
gamma_max = 200

def basicLinearTransform(img_original):
    res = cv.convertScaleAbs(img_original, alpha=alpha, beta=beta)
    img_corrected = cv.hconcat([img_original, res])
    #cv.imshow("Brightness and contrast adjustments", img_corrected)
    return img_corrected

def gammaCorrection(img_original):
    ## [changing-contrast-brightness-gamma-correction]
    lookUpTable = np.empty((1,256), np.uint8)
    for i in range(256):
        lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)

    res = cv.LUT(img_original, lookUpTable)
    ## [changing-contrast-brightness-gamma-correction]

    img_gamma_corrected = cv.hconcat([img_original, res])
    #cv.imshow("Gamma correction", img_gamma_corrected)
    return img_gamma_corrected

def on_linear_transform_alpha_trackbar(val):
    global alpha
    alpha = val / 100
    basicLinearTransform()

def on_linear_transform_beta_trackbar(val):
    global beta
    beta = val - 100
    basicLinearTransform()

def on_gamma_correction_trackbar(val):
    global gamma
    gamma = val / 100
    gammaCorrection()

##################################### MAIN #############################################
files = os.listdir(args.directory)
for file in files:
    if ".PNG" in file and '.txt' not in file.lower():
        logging.info(f'Processing {file}')
        #imgC = cv.imread(os.path.join(args.directory, file), cv.IMREAD_COLOR) #cv.IMREAD_GRAYSCALE)
        img = cv.imread(os.path.join(args.directory, file), cv.IMREAD_GRAYSCALE)
        
        # Matplotlib works but it is very slooooow.
        #time.sleep(1)
        #plt.imshow(img,'gray'),plt.title('ORIGINAL')
        #plt.draw()
        #plt.pause(0.001)
        # wait = input("Press the anykey to continue")
        #plt.close()
        
        logging.info(f'Native Image size: {np.shape(img)}')
        
        # Resize to something smaller so that the image fits on our screen
        #imS = cv.resize(img, (int(1936/2), int(1096/2))) 
        xsize = int(1548/2.5)
        ysize = int(1040/2.5)
        imS = cv.resize(img, (xsize, ysize)) 
        #imSC = cv.resize(imgC, (int(1548/2), int(1040/2))) 
        
        # Gamma correction
        img_gamma = gammaCorrection(imS)
        
        # Custom interpolated stretch function - works but not amazing
        #original = imS.copy()
        #xp = [0, 64, 128, 192, 255]
        #fp = [0, 128, 255, 255, 255]
        #x = np.arange(256)
        #table = np.interp(x, xp, fp).astype('uint8')
        #imS = cv.LUT(imS, table)
        #cv.imshow("original", original)
        #cv.imshow("stretched", imS)
        #cv.moveWindow('stretched', x=875, y=75)
        
        # https://docs.opencv.org/4.x/d5/daf/tutorial_py_histogram_equalization.html
        # Basic histogram equalization
        #equ = cv.equalizeHist(imS)
        #cv.imshow("original", imS)
        #cv.imshow("equalized_histogram", equ)
        #cv.moveWindow('equalized_histogram', x=875, y=75)
        
        # Contrast Limited Adaptive Histogram Equalization (tiled)
        # Color CLAHE: https://stackoverflow.com/questions/25008458/how-to-apply-clahe-on-rgb-color-images
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl1 = clahe.apply(imS)
        cl2 = cl1.copy()
        
        # Denoise
        #cv.fastNlMeansDenoisingMulti(cl1, 2, 5, cl2, 4, 7, 99) # meh
        # se=cv.getStructuringElement(cv.MORPH_RECT , (8,8))
        # Use dilation and opening to try to segment stars/objects - meh so far
        #kernel = np.ones((2,2),np.uint8)
        #dilation = cv.dilate(cl1,kernel,iterations = 1)
        #bg=cv.morphologyEx(dilation, cv.MORPH_OPEN, kernel)
        #cl2=cv.divide(cl1, bg, scale=255)
        #cl2=cv.threshold(cl2, 0, 255,  cv.THRESH_BINARY+cv.THRESH_OTSU )[1] 
        cl2 = cv.GaussianBlur(cl1, (13, 13), 0)
        cl2 = cv.Laplacian(cl2,cv.CV_8S, ksize=5)
        
        # Todo: implement a sequential spot tracker

        cv.imshow("original", cl1)
        
        k = cv.waitKey(0)
    else:
        logging.info(f'Ignoring {file}')