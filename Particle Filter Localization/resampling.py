'''
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

import numpy as np


class Resampling:
    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
    [Chapter 4.3]
    """
    def __init__(self):
        """
        Initialize resampling process parameters here
        """
        self._M = 100 #used for adaptively changing the number of particles

    def multinomial_sampler(self, X_bar):
        """
        param[in] X_bar : [num_particles x 4] sized array containing [x, y, theta, wt] values for all particles
        param[out] X_bar_resampled : [num_particles x 4] sized array containing [x, y, theta, wt] values for resampled set of particles
        """
        """
        TODO : Add your code here
        """
        X_bar_resampled =  np.zeros_like(X_bar)
        return X_bar_resampled

    def low_variance_sampler(self, X_bar):
        """
        param[in] X_bar : [num_particles x 4] sized array containing [x, y, theta, wt] values for all particles
        param[out] X_bar_resampled : [num_particles x 4] sized array containing [x, y, theta, wt] values for resampled set of particles
        """
       
        X_bar_resampled =  np.zeros_like(X_bar)
        num_particles, num_col = np.shape(X_bar)

        r = np.random.uniform(0, (1/num_particles))

        Wt = X_bar[:,3]
        # Normalize so that sum weights = 1. As U is always less than 1.Therefore sum of weights needs to be less than 1.
        Wt = Wt / np.sum(X_bar[:,3])
        
        i = 0
        c = Wt[0]
        for m in range(0, num_particles):
            U = r + m * (1/num_particles)
            while U > c:
                i += 1
                c = c + Wt[i]
            X_bar_resampled[m,:] = X_bar[i,:]

        return X_bar_resampled
