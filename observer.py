import numpy as np
import math

class Observer:
    """
    A simple observer that assumes targets move in straight lines at a constant speed.

    Take in full observation, return estimated full states and estimated velocities.

    Attributes: 
        observation_history    : emmmmm... 
        vel                    : velocities
        window                 : history length, i.e. history for how long before now we would like to keep
        step                   : discretization time step
        r_roi                  : Radius of region of interest
    """
    window = 5
    observation_history = [0]*window
    step = 0.1
    vel = 0
    states = 0
    r_roi = 1


    def __init__(self, x0, stp, windw, r=1):
        self.window = windw
        self.observation_history = [x0]*windw
        self.step = stp
        self.states = x0
        self.r_roi = r

    
    def feed(self, new_obsrv):
        self.observation_history.pop(0)    # Discard the oldest observation in the slot

        # # Screening observations based on distance (discard those that are far away)
        # self.states = np.array([new_obsrv[0]])
        # for obs in new_obsrv:
        #     distSqr = (obs[0] - new_obsrv[0][0])**2 + (obs[1] - new_obsrv[0][1])**2
        #     if distSqr <= self.r_roi**2:
        #         self.states = np.concatenate((self.states, [obs]), axis=0)
        self.observation_history.append(new_obsrv)

        # Successive difference method calculating velocities
        num_grp = math.floor(self.window/2)
        sum = self.observation_history[self.window-1] - self.observation_history[self.window-1]
        for i in range(1, num_grp+1):
            sum = sum + self.observation_history[self.window-i]
            sum = sum - self.observation_history[self.window-num_grp-i]
        self.states = new_obsrv
        self.vel = sum / num_grp / (self.step*num_grp)



## Test
if __name__ == '__main__':
    v = 1
    T = 0.01
    x = np.array([[0, 0], [1, 1]])
    observer = Observer(x, T, 5)
    for k in range(1, 10):
        x = x + v*T
        observer.feed(x)
        print(k)
        # print(observer.observation_history)
        print(observer.vel)
