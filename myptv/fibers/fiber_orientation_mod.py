# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 15:59:35 2022

@author: Eric Aschari

"""

import numpy as np
import math
from scipy import linalg


class FiberOrientation(object):
    '''A class to obtain the 3D fiber orientation from two fiber 
    points on every segmented image'''
    
    def __init__(self, X: np.ndarray,
                 B: np.ndarray):
        self.X = X # Center of fiber
        self.B = B # Endpoints of fiber
 

    def image2fiber(self, cams):
        '''
        input:
            cams: array of cameras
        output:
            cAvg: point on 3D line passing through fiber
            uAvg: 3D orientation of fiber
        '''
        
        cs = self.X
        bs = self.B
        # alphas = self.get_alphas()
        s = np.shape(cams)[0]
        c = np.zeros((3, math.comb(s,2)));
        u = np.zeros((3, math.comb(s,2)));
        r = 0 # non-nan result counter
        
        # for all camera pairs calculate the intersecting line
        for i in range(0,s-1):
            for j in range((i + 1),s):
                # get_r_ori???
                p1_n, p1_m = self.getPlane(np.transpose(np.array([cams[i].O])),
                              cams[i].get_r_ori(cs[i]),
                              cams[i].get_r_ori(bs[i]))
                p2_n, p2_m = self.getPlane(np.transpose(np.array([cams[j].O])),
                              cams[j].get_r_ori(cs[j]),
                              cams[j].get_r_ori(bs[j]))
                ctemp,utemp = self.intersectPlanes(p1_n,p1_m,p2_n,p2_m)
                
                if type(utemp) != str:
                    if type(utemp[0]) != 'NaN' and type(utemp[1]) != 'NaN' and type(utemp[2]) != 'NaN':
                        c[:,[r]] = ctemp
                        u[:,[r]] = utemp
                        r += 1
        
        cAvg,uAvg = self.averageLine(c,u)
        
        ori = self.get_ori(uAvg)
        
        return cAvg,uAvg,ori
    
    
    ### helper functions
    
    def solve_svd(self, A, b):
        '''
        input:
            A: matrix
            b: column vector
        output:
            x: vector so that A*x = b
        '''
        # compute svd of A
        U,s,Vh = linalg.svd(A)
        # U diag(s) Vh x = b <=> diag(s) Vh x = U.T b = c
        c = np.dot(U.T,b)
        # diag(s) Vh x = c <=> Vh x = diag(1/s) c = w (trivial inversion of a diagonal matrix)
        w = np.dot(np.diag(1/s),c)
        # Vh x = w <=> x = Vh.H w (where .H stands for hermitian = conjugate transpose)
        x = np.dot(Vh.conj().T,w)
        return x
    

    def getPlane(self, P1, P2, P3): 
        '''
        input:
            P123: three 3D points
        output:
            p_n: normal vector to plane spun by P123
            p_m: ax + by + cz = *m*
        '''
        A = np.array([[P1[0,0], P1[1,0], -1.0],
                      [P2[0,0], P2[1,0], -1.0],
                      [P3[0,0], P3[1,0], -1.0]])
        z = np.array([[-P1[2,0]], [-P2[2,0]], [-P3[2,0]]])
        # b = np.linalg.solve(A,z)
        # print('A___',A)
        # print('z___',z)
        b = self.solve_svd(A,z)
        
        p_n = np.array([[b[0,0]], [b[1,0]], [1]])
        p_m = b[2] / np.linalg.norm(p_n)
        p_n = p_n / np.linalg.norm(p_n)
        return p_n, p_m[0]


    def intersectPlanes(self, p1_n, p1_m, p2_n, p2_m): 
        '''
        input:
            p12_n: normal vectors of two planes
            p12_m: plane constant
        output:
            c,u: f(k) = c + k * u 
        '''
        n = np.concatenate([np.transpose(p1_n),np.transpose(p2_n)])
        if np.dot(np.transpose(p1_n),p2_n) > 0.9999:  #small angle between planes
            c = 'NaN'
            u = 'NaN'
            return c,u

        c = np.linalg.lstsq(n,np.array([[p1_m],[p2_m]]),rcond=-1)[0]
        u = linalg.null_space(n)
        return c,u
    

    def averageLine(self, c, u): 
        '''
        input:
            c,u: f(k) = c + k * u 
        output:
            cAvg: average of c
            uAvg: average of u
        '''
        A = np.zeros(np.shape(u))
        for i in range(0,np.shape(A)[1]):
            A[:,i] = u[:,0]
        
        s = np.sign(np.dot(np.transpose(A),u))[0]
        u = np.multiply(s,u)
        uAvg = np.mean(u,1)
        cAvg = np.mean(c,1)
        return cAvg,uAvg
    

    def get_alphas(self):
        '''
        input:
            self: centers and endpoints of fibers
        output:
            alphas: array of angles on camera planes
        '''
        alpha1 = np.arctan2((self.B[0,1] - self.X[0,1]), (self.B[0,0] - self.X[0,0]))[0]
        alpha2 = np.arctan2((self.B[1,1] - self.X[1,1]), (self.B[1,0] - self.X[1,0]))[0]
        # alpha3 = np.arctan2((self.B[2,1] - self.X[2,1]), (self.B[2,0] - self.X[2,0]))[0]
        # alpha4 = np.arctan2((self.B[3,1] - self.X[3,1]), (self.B[3,0] - self.X[3,0]))[0]
        
        # alphas = np.array([alpha1, alpha2, alpha3, alpha4])
        alphas = np.array([alpha1, alpha2])

        return alphas
    
    
    def get_ori(self,u):
        '''
        input:
            u: direction vector
        output:
            ori: angle array []
        '''
        xy_ori = np.arctan2(u[1],u[0])
        xz_ori = np.arctan2(u[2],u[0])
        ori = np.array([xy_ori,xz_ori])
        
        return ori
    
    
    
    
    

# ===============================================================
#      Fiber orientations from minimized projection err (Ron)
#         
#         (2 classes for the minimized projection method)
#
# Use fiber_traj_orientation to calculate fiber orientations and
# save the results on the disk. The fiber_ori_projection_method
# minimizes projection for single blobs. 
# ===============================================================


import numpy as np
from scipy.optimize import differential_evolution
import pandas as pd
from tqdm import tqdm


class fiber_ori_projection_method(object):
    '''
    A class for finding the orientation of a fiber by minimizing the 
    discrepancy with respect to the orientation of the blob.
    '''
    
    def __init__(self, cams, imageOrients, pos, ori0=None):
        '''
        cams - a list of Nc calibrated objects that can project lab-space 
               positions to image space positions.
        
        imageOrients - a list of Nc 2D unit vectors representing the 
                       orientation of the fiber long axis in image space of
                       each camra. to compare the results during minimization 
                       we make sure that the first vector component is always
                       positive.
                       
        pos - the lab space, 3D, position of the fiber center.
        
        ori0 - an optional initial guess for the orientation.
        '''
        
        self.cams = cams
        self.imgOri_lst = imageOrients
        
        for i in range(len(self.imgOri_lst)):
            
            # ensuring the format is right (first component positive)
            if self.imgOri_lst[i][0]<0:
                self.imgOri_lst[i] = self.imgOri_lst * -1
            
            # ensuring norm is 1
            self.imgOri_lst[i] = self.imgOri_lst[i] / np.linalg.norm(self.imgOri_lst[i])
            
        
        self.pos = pos
        
        if ori0 is None:
            self.ori0 = np.array([1,1,1])/3**0.5
            self.smartIG = 0 # index whether an initial guess was given
        
        else:
            self.ori0 = ori0
            self.smartIG = 1
        
    
    def OriToImageOri(self, ori):
        '''
        Given a vector, this function estimates and returns its 
        image space orientations in each camera, assuming it is centered
        at self.pos.
        '''
        ori = ori/np.linalg.norm(ori)
        ds = 1e-6
        p1 = self.pos + ori*ds/2
        p2 = self.pos - ori*ds/2
        
        imgOri_lst = []
        for e, cam in enumerate(self.cams):
            c1 = np.array(cam.projection(p1))[::-1]
            c2 = np.array(cam.projection(p2))[::-1]
            imgOri = (c1-c2) / np.linalg.norm(c1-c2)
            if imgOri[0]<0:
                imgOri = -1*imgOri
            imgOri_lst.append(imgOri)
            
        return imgOri_lst
    
    
    def Minimize_Ori(self):
        '''
        Searches for a vector that minimizes the difference in orientation 
        between its projection on image-space and segmented orientation of the
        blobs.
        
        returns
        
        ori - predicted orientation that minimizes projection vs. segmentation 
              error.
        
        MSE - mean squared orientation error.
        '''
        def ori_MSE(ori):
            res = np.array(self.OriToImageOri(ori)) - np.array(self.imgOri_lst)
            MSE = np.mean(np.sum(res**2, axis=0))
            return MSE
        
        # narrowing the search if an initial guess was given
        if self.smartIG: popsize = 5
        else: popsize = 15
        
        res_de = differential_evolution(ori_MSE, bounds=[(0,1),(-1,1),(-1,1)], popsize=popsize)
        ori = res_de.x / np.linalg.norm(res_de.x)
        
        MSE = res_de.fun
        
        return ori, MSE 
    
    
    



class fiber_traj_orientation(object):
    '''
    Given a trajectory file, a list of files with segmented 
    fiber directions, and a list of cameras, this class is
    used to determine the orientations of the fiber using
    the projection method.
    '''
    
    def __init__(self, traj_filename, blobs_ori_filename, cams):
        
        self.cams = cams
        
        self.blobs_ori = [] 
        for bfn in blobs_ori_filename:
            data = pd.read_csv(bfn, sep='\t', header=None)    
            self.blobs_ori.append(dict([(k,np.array(g)) for k,g in data.groupby(5)]))
        
        data = pd.read_csv(traj_filename, sep='\t', header=None)
        self.trajs = dict([(k,np.array(g)) for k, g in data.groupby(0)])
    
        
    def get_traj_orientation(self, traj):
        '''
        Finds the orientations of a given fiber by minimizing the
        error of its image projections

        traj - an array representing the trajectory

        blobs_ori - a list of dictionaries holding the image-space orientations
                    of blobs. the dictionaries' keys are frame numbers and the
                    list indexes are camera indexes.

        cams - a list of calibrated camera objects (also camer_wraper).
        '''
        blob_indexes = list(traj[:,4:-2])
        frames = list(traj[:,-1])
        traj_ori = []

        for e, frm in enumerate(frames):
            # get the image space orientation at frame e
            imageOrients = []
            for i in range(len(self.cams)):
                ind_ie = int(blob_indexes[e][i])
                imageOrients.append(self.blobs_ori[i][e][ind_ie,6:8])

            # set up an fiberOrientations instance
            pos = traj[e,1:4]
            if len(traj_ori)>0: ori0=traj_ori[-1]
            else: ori0=None
            FO = fiber_ori_projection_method(self.cams, imageOrients, pos, ori0=ori0)

            # Minimize for the fiber orientation
            traj_ori.append(FO.Minimize_Ori())

        return traj_ori
    
    
    def get_ori_lst(self):
        '''
        Iterates over the trajectories and obtains the orientation
        of each of them. The results are stored in self.ori_lst
        '''
        print('','Getting trajectory orientations...','')
        self.ori_lst = []
        
        for k in tqdm(self.trajs.keys()):
            traj = self.trajs[k]
            ori = self.get_traj_orientation(traj)
            for i in range(len(traj)):
                ln = traj[i].copy()
                ln[1:4] = ori[i][0]
                self.ori_lst.append(ln)
                
    
    def save_orientations(self, savename):
        '''
        Saves the results of self.get_ori_lst() as a tab separated 
        file with the same format as trajectories files.
        '''
        fmt = ['%d', '%.4f', '%.4f', '%.4f']
        for i in range(len(self.ori_lst[0])-6):
            fmt.append('%d')
        fmt += ['%.3f', '%.3f']
        np.savetxt(savename , self.ori_lst,
                delimiter='\t', fmt=fmt)
                
    
    
    
    
    
    
    
    
    
    
    
    
    
    