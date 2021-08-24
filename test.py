from numpy.random import gamma
import rps.robotarium as robotarium
# from rps.utilities.transformations import *
# from rps.utilities.barrier_certificates import *
# from rps.utilities.misc import *
# from rps.utilities.controllers import *
from clf_cbf_nmpc import CLF_CBF_NMPC,Simple_Catcher
import numpy as np
from observer import Observer
from estimator import Estimator



# N = 2
N = 50

a = np.array([[-1.2, -1.2 ,0]]).T
# a = np.array([[0, 0 ,0]]).T
d = np.array([[0.1, 0.35, -3.14]]).T # 
initial_conditions = a
for idx in range(1, N):
    initial_conditions = np.concatenate((initial_conditions, np.array([[2.6*np.random.rand()-1.0, 2.6*np.random.rand()-1.0, 6.28*np.random.rand()-3.14]]).T), axis=1)
print('Initial conditions:')
print(initial_conditions)

r = robotarium.Robotarium(number_of_robots=N, show_figure=True, initial_conditions=initial_conditions, sim_in_real_time=True)


def is_done(all_states):
    self_state = x[0]
    other_states = x[1:]

    # Check boundaries
    if(self_state[1]>1.5 or self_state[1]<-1.5 or self_state[0]>1.5 or self_state[0]<-1.5):
        print('Out of boundaries !!')
        return True
    # Reached goal?
    if (0.7<=self_state[0]<=1.5 and 0.7<=self_state[1]<=1.5):
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

obsrvr = Observer(x, 0.1, 6)

mpc_horizon = 10
T = 0.1
m_cbf = 8
m_clf = 0
gamma_k = 0.25
alpha_k = 0.1
clf_cbf_nmpc_solver = CLF_CBF_NMPC(mpc_horizon, T, m_cbf, m_clf, gamma_k, alpha_k)

while (is_done(x)==False):
    # print('\n----------------------------------------------------------')
    # print("Iteration %d" % times)

    x = r.get_poses().T

    # Observe & Predict

    obsrvr.feed(x)
    f = lambda x_, u_: x_-x_ + u_
    # print(obsrvr.vel[1:])
    estmtr = Estimator(x[1:], obsrvr.vel[1:], f, 0.1, 10)
    estmtr.predict()
    # print(estmtr.predict_states)
    global_states_sol, controls_sol, local_states_sol = clf_cbf_nmpc_solver.solve(x[0], [1.4, 1.4, 0], np.concatenate((np.array([obsrvr.states[1:]]), estmtr.predict_states), axis = 0))
    attacker_u = controls_sol[0]
    # attacker_u = np.array([0.2, 0.1])

    # defender_u = Simple_Catcher(x[0],x[1])

    dxu = np.zeros([N,2])
    dxu[0] = np.array([attacker_u[0],attacker_u[1]])

    for idx in range(1, N):
        # defender_u = Simple_Catcher(x[0],x[idx])
        # dxu[idx] = defender_u
        # dxu[idx] = np.array([0, 0]) 
        dxu[idx] = np.array([0.15, 0.1]) 
    # for idx in range(3, N)
    #     defender_u = Simple_Catcher(x[0],x[idx])
    #     dxu[idx] = defender_u
    #     dxu[idx] = np.array([0.2, 0.02]) 

    r.set_velocities(np.arange(N), dxu.T)

    times+=1
    i+=1
    r.step()
    # print('----------------------------------------------------------\n')

r.call_at_scripts_end()
