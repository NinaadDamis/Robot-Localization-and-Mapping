'''
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

import sys
import numpy as np
import math


class MotionModel:
    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
    [Chapter 5]
    """
    def __init__(self):
        """
        TODO : Tune Motion Model parameters here
        The original numbers are for reference but HAVE TO be tuned.
        """
        self._alpha1 = 0.00001
        self._alpha2 = 0.00001
        self._alpha3 = 0.0001
        self._alpha4 = 0.0001


    def update(self, u_t0, u_t1, x_t0):
        """
        param[in] u_t0 : particle state odometry reading [x, y, theta] at time (t-1) [odometry_frame]
        param[in] u_t1 : particle state odometry reading [x, y, theta] at time t [odometry_frame]
        param[in] x_t0 : particle state belief [x, y, theta] at time (t-1) [world_frame]
        param[out] x_t1 : particle state belief [x, y, theta] at time t [world_frame]
        """
        #odometry reading vecctor at time (t)
        xprime_bar = u_t1[0]
        yprime_bar = u_t1[1]
        theta_prime_bar = u_t1[2]

        #odometry reading vector at time (t-1)
        x_bar = u_t0[0]
        y_bar = u_t0[1]
        theta_bar = u_t0[2]

        #particle state belief vector at time (t-1)
        x = x_t0[:,0]
        y = x_t0[:,1]
        theta = x_t0[:,2]

        #implement odometry motion model algorithm
        y_diff = yprime_bar - y_bar
        x_diff = xprime_bar - x_bar

        y_diff_sq = (y_bar - yprime_bar)**2
        x_diff_sq = (x_bar - xprime_bar)**2

        delta_rot1 = (math.atan2(y_diff, x_diff)) - theta_bar
        delta_trans = math.sqrt(x_diff_sq + y_diff_sq)
        delta_rot2 = theta_prime_bar - theta_bar - delta_rot1

        var1 = (self._alpha1 * (delta_rot1**2)) + (self._alpha2 * (delta_trans**2))
        var2 = (self._alpha3 * (delta_trans**2)) + (self._alpha4 * (delta_rot1**2)) + (self._alpha4 * (delta_rot2**2))
        var3 = (self._alpha1 * (delta_rot2**2)) + (self._alpha2 * (delta_trans**2))

        delta_rot1_hat = delta_rot1 - np.random.normal(0, np.math.sqrt(var1))
        delta_trans_hat = delta_trans - np.random.normal(0, np.math.sqrt(var2))
        delta_rot2_hat = delta_rot2 - np.random.normal(0, np.math.sqrt(var3))

        xprime = x + delta_trans_hat * np.cos(theta + delta_rot1_hat)
        yprime = y + delta_trans_hat * np.sin(theta + delta_rot1_hat)
        theta_prime = theta + delta_rot1_hat + delta_rot2_hat

        x_t1 = np.transpose(np.array([xprime, yprime, theta_prime]))

        return x_t1
