# -*- coding: utf-8 -*-
#!/usr/bin/env python3

"""
Created on Sun Dec 28 12:00:44 2025

@author: ron



A script to generate synthetic images of fibers to test MyFTV functionalities

"""



from myptv.imaging_mod import camera_wrapper
from PIL import Image, ImageDraw
import numpy as np
from numpy.random import uniform



# =====================
# Loading cameras:
# =====================

camNames = ['cam1', 'cam2', 'cam3']
cams = [camera_wrapper(cn, '.') for cn in camNames]
for cam in cams:
    cam.load()
    


    
# =====================
# Parameters:
# =====================

N_fibers = 5               # number of fibers
Length = 5                  # mm
xmin, xmax = 0, 100         # x ROI
ymin, ymax = 5, 105         # y ROI
zmin, zmax = -50, 0.0       # z ROI
resolution = 2048, 2048     # image pixel resolution

n = 50                      # number of points making up a fiber
s = 2                       # sigma of a gaussian intensity point
I = 40                      # intensity of a fiber making point




# =====================
# Generating the image:
# =====================

radius = int(3*s)
x_ = range(-radius, radius+1)
X, Y = np.meshgrid(x_, x_)
blob_image = I * np.exp(-( ((X)**2 + (Y)**2) /2/s**2))

images = [np.zeros((resolution[1], resolution[0])) for cam in cams]
pos_lst = []
ori_lst = []
for i in range(N_fibers):
    
    # random fiber center position
    X = uniform(xmin, xmax), uniform(ymin, ymax), uniform(zmin, zmax)
    pos_lst.append(X)
    
    # vector along the fiber
    R = np.random.uniform(-0.5,0.5, size=3)
    R = R / np.linalg.norm(R) * Length
    z_factor = 1 + (abs(R[2]) / Length)  # <- scaling the intensity based on z
    ori_lst.append(R / Length)
    
    for e, cam in enumerate(cams):
        img = images[e]
        
        X_lst = X + np.dot(R.reshape((3,1)), [np.linspace(-0.5, 0.5, num=n)]).T
        
        for x in X_lst:
            cx, cy = np.array(cam.projection(x)).astype('int')
            i0, i1, j0, j1 = cy-radius, cy+radius+1, cx-radius, cx+radius+1
            img[i0:i1, j0:j1] += blob_image / z_factor
    
    

for e, cam in enumerate(cams):
    img = images[e]
    img[img>255] = 255
    img = img.astype('int8')
    
    pil_im = Image.fromarray(img, mode='L')
    pil_im.save('im_cam%d.tif'%(e+1))
    
    
np.savetxt('ground_truth_pos', pos_lst, delimiter='\t', fmt='%.3f')
np.savetxt('ground_truth_ori', ori_lst, delimiter='\t', fmt='%.4f')




