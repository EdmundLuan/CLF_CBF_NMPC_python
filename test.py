import rps.robotarium as robotarium
from rps.utilities.transformations import *
from rps.utilities.barrier_certificates import *
from rps.utilities.misc import *
from rps.utilities.controllers import *
from clf_cbf_nmpc import CLF_CBF_NMPC,Base_NMPC,Simple_Catcher
import numpy as np

N = 2

a = np.array([-1.2, -0.2 ,0]).T
d = np.array([0.1, 0.35, -3.14 ]).T
initial_conditions = np.array([a,d]).T 
r = robotarium.Robotarium(number_of_robots=N, show_figure=True, initial_conditions=initial_conditions,sim_in_real_time=True)

def is_done(a_state,d_state):
    distance = np.sqrt(np.sum(np.square(a_state[:2] - d_state[:2])))

    if (distance<0.2 or a_state[1]>1.5 or a_state[1]<-1.5 ):
        done_n = True
    else:
        done_n = False
    
    if (0.5<=a_state[0]<=1.5 and 0.5<=a_state[1]<=1.5):
        done_n = True
    return done_n


x = r.get_poses().T
r.step()

i=0
times = 0

while (is_done(x[0],x[1])==False):

    x = r.get_poses().T
    
    # attacker_u = Base_NMPC(x[0],x[1])
    attacker_u = CLF_CBF_NMPC(x[0],x[1])

    defender_u = Simple_Catcher(x[0],x[1])

    dxu = np.zeros([2,2])
    dxu[0] = np.array([attacker_u[0],attacker_u[1]])
    dxu[1] = np.array([defender_u[0],defender_u[1]])

    r.set_velocities(np.arange(N), dxu.T)

    times+=1
    i+=1
    r.step()

r.call_at_scripts_end()
