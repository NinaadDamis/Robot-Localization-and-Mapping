'''
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

import numpy as np
import math
import time
from matplotlib import pyplot as plt

from map_reader import MapReader


class SensorModel:
    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
    [Chapter 6.3]
    """
    def __init__(self, occupancy_map):
        """
        TODO : Tune Sensor Model parameters here
        The original numbers are for reference but HAVE TO be tuned.
        """
        self._z_hit = 1
        self._z_short = 0.1
        self._z_max = 0.1
        self._z_rand = 150 #100

        self._sigma_hit = 100 #50
        self._lambda_short = 0.1

        # Used in p_max and p_rand, optionally in ray casting
        self._max_range = 1000

        # Used for thresholding obstacles of the occupancy map
        self._min_probability = 0.35

        # Used in sampling angles in ray casting
        self._subsampling = 5

        # Sensor displacement value
        self._sensor_distance = 25

        # Occupancy map
        self._map = occupancy_map

        # Ray casting step size
        self._step = 10 # Step oncecell size at a a time

        # Bool to use log likelihood
        self._use_log_likelihood = False

    def normal_distribution(self,x,mu,sd):
        var = np.power(sd,2)
        return np.exp(- np.power(x - mu,2) / (2 * var)) / np.sqrt(2*math.pi*var)

    def subsampling(self,z_t1_arr,x_t1):

        num_particles = z_t1_arr.shape[0]
        angle = x_t1[:,2].reshape((num_particles,1))
        angle_arr = np.linspace(-90 , 90 ,180) * (np.pi/ 180) 
        angle_arr = np.tile(angle_arr,(num_particles,1))
        angle_arr = angle_arr + angle # Add robot theta to get angles in world frame

        # Get subsampled arrays
        subsampled_z_arr  = z_t1_arr[:,0:-1:self._subsampling]
        subsampled_angle_arr = angle_arr[:,0:-1:self._subsampling]
        return subsampled_z_arr, subsampled_angle_arr

    def ray_casting(self,z_t1_arr,angle_arr,x_t1):

        num_particles = z_t1_arr.shape[0]
        num_angles    = z_t1_arr.shape[1]
        angle = x_t1[:,2]
        # Account for sensor offset to get position of laser in world frame.
        x = x_t1[:,0] + self._sensor_distance * np.cos(angle)
        y = x_t1[:,1] + self._sensor_distance * np.sin(angle)

        arr = np.zeros(z_t1_arr.shape)
        ray_cast_x = np.zeros(z_t1_arr.shape)
        ray_cast_y = np.zeros(z_t1_arr.shape)
        # TODO
        for m in range(num_particles):
            xx = x[m]
            yy = y[m]
            for i in range(num_angles):
                # Get ray casting distance
                z = z_t1_arr[m,i]
                a = angle_arr[m,i]
                xd = x[m]
                yd = y[m]
                xd_idx = round(xd/10)
                yd_idx = round(yd/10)
                ray_dist = 0
                # Check if index within map, not an obstacle and within max range
                while (0 <= xd_idx < 800) and (0 <= yd_idx < 800) and (self._map[int(yd_idx),int(xd_idx)] <= self._min_probability) and (ray_dist <= self._max_range):
                    xd += self._step * np.cos(a)
                    yd += self._step * np.sin(a)
                    xd_idx = round(xd/10)
                    yd_idx = round(yd/10)
                    ray_dist = np.sqrt(np.power(xd - xx,2) + np.power(yd - yy,2))
                arr[m,i] = ray_dist
                ray_cast_x[m,i] = xd_idx
                ray_cast_y[m,i] = yd_idx

        return arr, ray_cast_x, ray_cast_y

    def calculate_measurement_likelihood(self, z_t1_arr,z_tstar_arr):

        # Phit
        sigma_hit_arr = self._sigma_hit * np.ones(np.shape(z_t1_arr))
        Phit_arr = np.array(list(map(self.normal_distribution,z_t1_arr,z_tstar_arr,sigma_hit_arr)))
        Phit = np.where((z_t1_arr >= 0) & (z_t1_arr <= self._max_range), Phit_arr, 0)

        # Pshort
        Pshort_arr = self._lambda_short * np.exp(-self._lambda_short* z_t1_arr)
        Pshort = np.where((z_t1_arr >= 0) & (z_t1_arr <= z_tstar_arr), Pshort_arr, 0)

        # Pmax
        Pmax = np.where(z_t1_arr >= self._max_range,1,0)

        # Prand
        Prand = np.where((z_t1_arr >= 0) & (z_t1_arr < self._max_range), 1/ self._max_range, 0)

        # Weighted Sum
        P = self._z_hit * Phit + self._z_short * Pshort + self._z_max * Pmax + self._z_rand * Prand

        # If log odds representation
        if self._use_log_likelihood:
            P = P / (1- P)
            P = np.log(P)
            P = np.sum(P,axis=1)
        # Else Product of individual probabilities (independent assumption)
        else:
            P = np.prod(P,axis = 1)

        return P


    def beam_range_finder_model(self, z_t1_arr, x_t1):
        """
        param[in] z_t1_arr : laser range readings [array of 180 values] at time t
        param[in] x_t1 : particle state belief [x, y, theta] at time t [world_frame]
        param[out] prob_zt1 : likelihood of a range scan zt1 at time t
        """
        """
        TODO : Add your code here
        """
        z_arr, angles_arr = self.subsampling(z_t1_arr,x_t1)
        z_tstar_arr, ray_cast_x, ray_cast_y = self.ray_casting(z_arr,angles_arr,x_t1)
        prob_zt1 = self.calculate_measurement_likelihood(z_arr,z_tstar_arr)

        return prob_zt1, ray_cast_x, ray_cast_y, z_tstar_arr 
