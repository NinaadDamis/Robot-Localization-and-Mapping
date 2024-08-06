'''
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

import argparse
import numpy as np
import sys, os

from map_reader import MapReader
from motion_model import MotionModel
from sensor_model import SensorModel
from resampling import Resampling

from matplotlib import pyplot as plt
from matplotlib import figure as fig
import time


def visualize_map(occupancy_map):
    fig = plt.figure()
    mng = plt.get_current_fig_manager()
    plt.ion()
    plt.imshow(occupancy_map, cmap='Greys')
    plt.axis([0, 800, 0, 800])


def visualize_timestep(X_bar, tstep, output_path):
    x_locs = X_bar[:, 0] / 10.0
    y_locs = X_bar[:, 1] / 10.0
    scat = plt.scatter(x_locs, y_locs, c='r', marker='o')
    plt.savefig('{}/{:04d}.png'.format(output_path, tstep))
    plt.pause(0.00001)
    scat.remove()

def visualize_raycasting(X_bar,x_t0, tstep, output_path, ray_cast_x, ray_cast_y, z_star_arr):
    print("Visualizing timestep")
    x_locs = X_bar[:, 0] / 10.0
    y_locs = X_bar[:, 1] / 10.0
    for i in range(len(ray_cast_x)):
        print("Line dist = ", z_star_arr[i])
        xx = [x_t0[0]/10.0 , ray_cast_x[i]]
        yy = [x_t0[1]/10.0 , ray_cast_y[i]]
        plt.plot(xx, yy, 'bo', linestyle="--")
    plt.pause(0.1)


def init_particles_random(num_particles, occupancy_map):

    # initialize [x, y, theta] positions in world_frame for all particles
    y0_vals = np.random.uniform(0, 7000, (num_particles, 1))
    x0_vals = np.random.uniform(3000, 7000, (num_particles, 1))
    theta0_vals = np.random.uniform(-3.14, 3.14, (num_particles, 1))

    # initialize weights for all particles
    w0_vals = np.ones((num_particles, 1), dtype=np.float64)
    w0_vals = w0_vals / num_particles

    X_bar_init = np.hstack((x0_vals, y0_vals, theta0_vals, w0_vals))

    return X_bar_init


def init_particles_freespace(num_particles, occupancy_map):

    # initialize [x, y, theta] positions in world_frame for all particles
    """
    TODO : Add your code here
    This version converges faster than init_particles_random
    """
    y_idxs, x_idxs = np.where(occupancy_map == 0) # Free cells with P = 0
    idxs = np.random.choice(len(x_idxs) - 1 , num_particles) # Select n random indices of x_idxs, y_idxs 
    x0_vals     = x_idxs[idxs]*10 + 5 # Adding 5 to get centres of cells ?
    y0_vals     = y_idxs[idxs]*10 + 5 # Multiplying by 10 to get cm
    x0_vals     = np.array(x0_vals).reshape((num_particles,1))
    y0_vals     = np.array(y0_vals).reshape((num_particles,1))
    theta0_vals = np.random.uniform(-3.14, 3.14, (num_particles, 1))
    
    # initialize weights for all particles
    w0_vals = np.ones((num_particles, 1), dtype=np.float64)
    w0_vals = w0_vals / num_particles
    X_bar_init = np.hstack((x0_vals, y0_vals, theta0_vals, w0_vals))

    return X_bar_init


if __name__ == '__main__':
    """
    Description of variables used
    u_t0 : particle state odometry reading [x, y, theta] at time (t-1) [odometry_frame]
    u_t1 : particle state odometry reading [x, y, theta] at time t [odometry_frame]
    x_t0 : particle state belief [x, y, theta] at time (t-1) [world_frame]
    x_t1 : particle state belief [x, y, theta] at time t [world_frame]
    X_bar : [num_particles x 4] sized array containing [x, y, theta, wt] values for all particles
    z_t : array of 180 range measurements for each laser scan
    """
    """
    Initialize Parameters
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_map', default='../data/map/wean.dat') 
    parser.add_argument('--path_to_log', default='../data/log/robotdata1.log') 
    parser.add_argument('--output', default='results')
    parser.add_argument('--num_particles', default=500, type=int)
    parser.add_argument('--viz_raycasting', default=False)#action='store_true'
    parser.add_argument('--visualize', default=True )#action='store_true'
    parser.add_argument('--test_motion_model', default=False)
    args = parser.parse_args()

    src_path_map = args.path_to_map
    src_path_log = args.path_to_log
    os.makedirs(args.output, exist_ok=True)

    map_obj = MapReader(src_path_map)
    occupancy_map = map_obj.get_map()
    logfile = open(src_path_log, 'r')

    motion_model = MotionModel()
    sensor_model = SensorModel(occupancy_map)
    resampler = Resampling()

    num_particles = args.num_particles
    viz_raycasting = args.viz_raycasting
    test_motion_model = args.test_motion_model
    X_bar = init_particles_freespace(num_particles, occupancy_map)
    if(test_motion_model):
        X_bar = np.array([[4.7650000e+03,3.8750000e+03,-1.2315289e+00,1.0000000e+00]])
        num_particles = 1

    """
    Monte Carlo Localization Algorithm : Main Loop
    """
    if args.visualize:
        visualize_map(occupancy_map)

    first_time_idx = True
    for time_idx, line in enumerate(logfile):

        # Read a single 'line' from the log file (can be either odometry or laser measurement)
        # L : laser scan measurement, O : odometry measurement
        meas_type = line[0]

        # convert measurement values from string to double
        meas_vals = np.fromstring(line[2:], dtype=np.float64, sep=' ')

        # odometry reading [x, y, theta] in odometry frame
        odometry_robot = meas_vals[0:3]
        time_stamp = meas_vals[-1]

        # ignore pure odometry measurements for (faster debugging)
        # if ((time_stamp <= 0.0) | (meas_type == "O")):
        #     continue

        if (meas_type == "L"):
            # [x, y, theta] coordinates of laser in odometry frame
            odometry_laser = meas_vals[3:6]
            # 180 range measurement values from single laser scan
            ranges = meas_vals[6:-1]

        print("Processing time step {} at time {}s".format(
            time_idx, time_stamp))

        if first_time_idx:
            u_t0 = odometry_robot
            first_time_idx = False
            continue

        X_bar_new = np.zeros((num_particles, 4), dtype=np.float64)
        u_t1 = odometry_robot

        # Note: this formulation is intuitive but not vectorized; looping in python is SLOW.
        # Vectorized version will receive a bonus. i.e., the functions take all particles as the input and process them in a vector.
        # for m in range(0, num_particles):
        #     """
        #     MOTION MODEL
        #     """
        #     x_t0 = X_bar[m, 0:3]
        #     x_t1 = motion_model.update(u_t0, u_t1, x_t0)
        #     if test_motion_model:
        #         X_bar_new[m, :] = np.hstack((x_t1, X_bar[m, 3]))
        #     else: # SENSOR MODEL
        #         if (meas_type == "L"):
        #             z_t = ranges
        #             w_t, ray_cast_x, ray_cast_y, z_star_arr = sensor_model.beam_range_finder_model(z_t, x_t1)
        #             X_bar_new[m, :] = np.hstack((x_t1, w_t))
        #             if(viz_raycasting):
        #                 visualize_raycasting(X_bar,x_t0, time_idx, args.output, ray_cast_x, ray_cast_y, z_star_arr)
        #         else:
        #             X_bar_new[m, :] = np.hstack((x_t1, X_bar[m, 3]))

        # X_bar = X_bar_new
        # u_t0 = u_t1

        #vectorized model

        x_t0 = X_bar[:,0:3]
        start = time.time()
        x_t1 = motion_model.update(u_t0, u_t1, x_t0)
        if test_motion_model:
            X_bar_new = np.append(x_t1, X_bar[:,3].reshape((num_particles, 1)), axis=1)
        else:
            if (meas_type == "L"):
                z_t = ranges
                z_t = np.tile(z_t,(num_particles,1))
                w_t, ray_cast_x, ray_cast_y, z_star_arr = sensor_model.beam_range_finder_model(z_t, x_t1)
                w_t = w_t.reshape((num_particles,1))
                X_bar_new = np.hstack((x_t1, w_t))
                if (viz_raycasting):
                    visualize_raycasting(X_bar, x_t0, time_idx, args.output, ray_cast_x, ray_cast_y, z_star_arr)
            else:
                X_bar_new = np.append(x_t1, X_bar[:,3].reshape((num_particles, 1)), axis=1)
        
        X_bar = X_bar_new
        u_t0 = u_t1

        """
        RESAMPLING
        """
        if not test_motion_model:
            X_bar = resampler.low_variance_sampler(X_bar)
        end = time.time()
        tt = end - start
        if(viz_raycasting and meas_type=="L"):
            print("Pausing after raycast viz as not able to remove prev drawn lines.")
            plt.pause(5)
        if args.visualize:
            visualize_timestep(X_bar, time_idx, args.output)
