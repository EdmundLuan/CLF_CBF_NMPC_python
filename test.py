import rps.robotarium as robotarium
# from rps.utilities.transformations import *
# from rps.utilities.barrier_certificates import *
# from rps.utilities.misc import *
# from rps.utilities.controllers import *
from clf_cbf_nmpc import CLF_CBF_NMPC,Base_NMPC,Simple_Catcher
import numpy as np

# N = 2
N = 15

a = np.array([[-1.2, -1.2 ,0]]).T
# d = np.array([[0.1, 0.35, -3.14]]).T # 
initial_conditions = a
for idx in range(N-1):
    initial_conditions = np.concatenate((initial_conditions, np.array([[1.7*np.random.rand()-0.5, 1.7*np.random.rand()-0.5, 6.28*np.random.rand()-3.14]]).T), axis=1)
print(initial_conditions)

r = robotarium.Robotarium(number_of_robots=N, show_figure=True, initial_conditions=initial_conditions,sim_in_real_time=True)


def is_done(all_states):
    self_state = x[0]
    other_states = x[1:]

    # Check boundaries
    if(self_state[1]>1.5 or self_state[1]<-1.5 or self_state[0]>1.5 or self_state[0]<-1.5):
        print('Out of boundaries !!')
        return True
    # Reached goal?
    if (0.5<=self_state[0]<=1.5 and 0.5<=self_state[1]<=1.5):
        print('Reach goal successfully!')
        return True

    for idx in range(np.size(other_states, 0)):
        # if(other_states[idx][0]>1.5 or other_states[idx][0]<-1.5 or other_states[idx][1]>1.5 or other_states[idx][1]<-1.5 ):
        #     print('Vehicle %d is out of boundaries !!' % idx+1)
        #     return True
        distSqr = (self_state[0]-other_states[idx][0])**2 + (self_state[1]-other_states[idx][1])**2
        if distSqr < (0.2)**2:
            print('Get caught, mission failed !')
            return True
    
    return False


x = r.get_poses().T
r.step()

i=0
times = 0

while (is_done(x)==False):

    x = r.get_poses().T
    
    # attacker_u = Base_NMPC(x[0],x[1:])
    attacker_u = CLF_CBF_NMPC(x[0], x[1:])
    # attacker_u = np.array([0.2, 0.1])

    # defender_u = Simple_Catcher(x[0],x[1])

    dxu = np.zeros([N,2])
    dxu[0] = np.array([attacker_u[0],attacker_u[1]])

    for idx in range(1, N):
        defender_u = Simple_Catcher(x[0],x[idx])
        dxu[idx] = defender_u
        dxu[idx] = np.array([0.2, 0.3]) 
    # for idx in range(3, N):
    #     defender_u = Simple_Catcher(x[0],x[idx])
    #     dxu[idx] = defender_u
    #     dxu[idx] = np.array([0.2, 0.02]) 

    r.set_velocities(np.arange(N), dxu.T)

    times+=1
    print("Iteration %d" % times)
    i+=1
    r.step()

r.call_at_scripts_end()
