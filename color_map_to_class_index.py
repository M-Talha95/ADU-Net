import cv2
import sys
import os.path
import numpy as np

def apply_GID_colormap(image):
    
        '''
	label information of 5 classes:
built-up 
RGB:    255,    0,    0
farmland
RGB:    0,    255,    0
forest
RGB:  0,    255,  255
meadow
RGB:  255, 255,     0
water
RGB:  0,        0,  255
'''

        img = np.zeros((image.shape[0], image.shape[1], 3), dtype = np.uint8)
        
	# indices = np.where(image == 5)                  #Background(Black)
        # img[indices] = (0,    0,    0)
        
        indices = np.where(image == 1)                  #Built-Up(RED)
        img[indices] = (255,    0,    0)

        indices = np.where(image == 2)                  #Farmland(Green)
        img[indices] = (0,    255,    0)

        indices = np.where(image == 3)                  #Forest(Sky Blue)
        img[indices] = (0,    255,  255)

        indices = np.where(image ==4)                  #Meadow(Yellow)
        img[indices] = (255, 255,     0)

        indices = np.where(image == 5)                  #Water(Blue)
        img[indices] = (0,        0,  255)

        indices = np.where(image == 0)                  #Background(Black)
        img[indices] = (0,    0,    0)

        return img
    