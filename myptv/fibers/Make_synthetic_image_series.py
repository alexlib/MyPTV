# -*- coding: utf-8 -*-
#!/usr/bin/env python3

"""
Created on Mon Dec 29 14:22:05 2025

@author: ron

A code to generate a series of synthetic images with
fibers, based on given trajectories.
"""

from myptv.imaging_mod import camera_wrapper
from PIL import Image, ImageDraw
import numpy as np
from numpy.random import uniform





# =========================================================
# Constructing fiber position and orientation trajectories:
# =========================================================


x0_lst = [[10,10,0], 
          [25,25,-10],
          [50,50,-20],
          [75,75,-30],
          [90,90,-40]]

v_lst = [[1,0,0],
         [-1,1,0],
         [0,0,-1],
         [0,-1,1],
         [0,-1,0]]



ori_0 = [[0.0, 30.0],
         [45.0, 25.0],
         [130.0, 10.0],
         [0.0, 90.0],
         [70.0, 90.0],
         [50.0, 150.0]]

ori_velocity = [[5.0, 0.0],
                [-5.0, 0.0],
                [0.0, 5.0],
                [0.0, -5.0],
                [15.0, 0.0],
                [-15.0, 0.0]]
         

N_fibers = len(x0_lst)
N_frames = 5
traj = []
ori_traj = []

for i in range(N_fibers):
    dx = np.dot(np.array(v_lst[i]).reshape(3,1), [np.arange(N_frames)]).T
    traj.append(np.array(x0_lst[i]).reshape((1,3)) + dx)
    
    theta = (ori_0[i][0] + np.arange(N_frames)*ori_velocity[i][0])/180.0*np.pi
    phi = (ori_0[i][1] + np.arange(N_frames)*ori_velocity[i][1])/180.0*np.pi
    ori = [np.sin(phi)*np.cos(theta), np.sin(theta)*np.sin(phi), np.cos(theta)]
    ori_traj.append(np.array(ori).T)
    
    if i==3: print(phi)



# =====================
# Loading cameras:
# =====================

camNames = ['cam1', 'cam2', 'cam3']
cams = [camera_wrapper(cn, '.') for cn in camNames]
for cam in cams:
    cam.load()




# ======================
# Generating the images:
# ======================

Length = 5                  # mm
resolution = 2048, 2048     # image pixel resolution
n = 50                      # number of points making up a fiber
s = 2                       # sigma of a gaussian intensity point
I = 40                      # intensity of a fiber making point

radius = int(3*s)
x_ = range(-radius, radius+1)
X, Y = np.meshgrid(x_, x_)
blob_image = I * np.exp(-( ((X)**2 + (Y)**2) /2/s**2))


for k in range(N_frames):
    
    images = [np.zeros((resolution[1], resolution[0])) for cam in cams]
    pos_lst = []
    ori_lst = []
        
    for i in range(N_fibers):
        
        # random fiber center position
        X = traj[i][k]
        pos_lst.append(X)
        
        # vector along the fiber
        R = ori_traj[i][k]
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
            pil_im.save('./cam%d_images/im%03d.tif'%(e+1,k+1))
    
    
    np.savetxt('./ground_truth/ground_truth_pos_im_%02d'%(k+1), pos_lst, delimiter='\t', fmt='%.3f')
    np.savetxt('./ground_truth/ground_truth_ori_im_%02d'%(k+1), ori_lst, delimiter='\t', fmt='%.4f')
