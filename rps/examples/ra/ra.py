import rps.robotarium as robotarium
from rps.utilities.transformations import *
from rps.utilities.barrier_certificates import *
from rps.utilities.misc import *
from rps.utilities.controllers import *

import numpy as np
import time

def compute_a(states):
    state = states
    return np.array([0,0])


## notice theta [-pi,pi]
def compute_d(states):
    u = np.zeros(2)
    u[0] = 0
    u[1] = 1
    print(states[1,2])
    return u

# Instantiate Robotarium object
N = 2
# initial_conditions = np.array(np.mat('1 0.5; 0.8 -0.3 ; 0 0 '))
initial_conditions = np.array([[-1.5,0,0],[1,0,-np.pi]]).T 
r = robotarium.Robotarium(number_of_robots=N, show_figure=True, initial_conditions=initial_conditions,sim_in_real_time=True)

# Define goal points by removing orientation from poses
goal_points = np.array(np.mat('0.5 1; -0.3 0.8 ; -0.6 0 '))

# generate_initial_conditions(N)

# Create unicycle pose controller
unicycle_pose_controller = create_clf_unicycle_pose_controller()

# Create barrier certificates to avoid collision
uni_barrier_cert = create_unicycle_barrier_certificate()

# define x initially
x = r.get_poses().T
r.step()

# While the number of robots at the required poses is less
# than N...
while (np.size(at_pose(x.T, goal_points)) != N):

    # Get poses of agents
    x = r.get_poses().T
    
    # print(x)
    # Create unicycle control inputs
    # dxu = unicycle_pose_controller(x.T, goal_points)
    # print(dxu)
    # # Create safe control inputs (i.e., no collisions)
    # dxu = uni_barrier_cert(dxu, x.T)
    dxu = np.zeros([2,2])
    dxu[0] = compute_a(x)
    dxu[1] = compute_d(x)
    # # Set the velocities
    r.set_velocities(np.arange(N), dxu.T)
    # r.set_velocities(np.array([0,1]), np.array([[0.1,-0.1],[0.1,-0.1]]) )
    # Iterate the simulation
    r.step()

#Call at end of script to print debug information and for your script to run on the Robotarium server properly
r.call_at_scripts_end()
