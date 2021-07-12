import numpy as np

from cvxopt import matrix
from cvxopt.blas import dot
from cvxopt.solvers import qp, options
from cvxopt import matrix, sparse

import casadi as ca
import numpy as np

def CLF_CBF_NMPC(attacker_state,defender_state):
    
    opti = ca.Opti()

    ## parameters
    T = 0.05
    N = 10
    M_CBF = 4
    M_CLF = 3
    v_max = 0.5
    omega_max = 2
    safe_distance = 0.4
    Q = np.array([[1.0, 0.0, 0.0],[0.0, 1.0, 0.0],[0.0, 0.0, 0.0]])
    R = np.array([[0.5, 0.0], [0.0, 0.001]])
    W_slack = np.array([[1000]])
    goal = np.array([[1.2, 1.2, 0.0]])

    ## control variables, linear velocity v and angle velocity omega
    opt_x0 = opti.parameter(3)
    opt_controls = opti.variable(N, 2)
    v = opt_controls[:, 0]
    omega = opt_controls[:, 1]

    ## state variables
    opt_states = opti.variable(N+1, 3)
    x = opt_states[:, 0]
    y = opt_states[:, 1]
    theta = opt_states[:, 2]

    ## slack variable to avoid infeasibility between CLF and CBF constraints
    d_slack = opti.variable(N, 1)

    ## create funciont for F(x), H(x) and V(x)
    f = lambda x_, u_: ca.vertcat(*[u_[0]*ca.cos(x_[2]), u_[0]*ca.sin(x_[2]), u_[1]])
    h = lambda x_: (x_[0] - defender_state[0]) ** 2 + (x_[1] - defender_state[1]) ** 2 - safe_distance**2 
    V = lambda x_: (x_[0] - 1.2) ** 2 + (x_[1] - 1.2) ** 2 

    ## init_condition
    opti.subject_to(opt_states[0, :] == opt_x0.T)
    opti.subject_to(opti.bounded(-1.45, x, 1.45))
    opti.subject_to(opti.bounded(-1.45, y, 1.45))
    opti.subject_to(opti.bounded(-v_max, v, v_max))
    opti.subject_to(opti.bounded(-omega_max, omega, omega_max))   

    ## system model constrain
    for i in range(N):
        x_next = opt_states[i, :] + f(opt_states[i, :], opt_controls[i, :]).T*T
        opti.subject_to(opt_states[i+1, :]==x_next)
    
    ## CBF constrain
    for i in range(M_CBF):
        opti.subject_to(h(opt_states[i+1, :]) >= (1-0.1)*h(opt_states[i, :]) ) 

    ## CLF constrain
    for i in range(M_CLF):
        opti.subject_to(V(opt_states[i+1, :]) <= (1-0.1)*V(opt_states[i, :]) + d_slack[i, :]) ## CLF constrain

    #### cost function
    obj = 0 
    for i in range(N):
        obj = obj + ca.mtimes([(opt_states[i, :] - goal), Q, (opt_states[i, :]- goal).T]) + ca.mtimes([opt_controls[i, :], R, opt_controls[i, :].T]) + ca.mtimes([d_slack[i, :], W_slack, d_slack[i, :].T]) 

    opti.minimize(obj)
    opts_setting = {'ipopt.max_iter':100, 'ipopt.print_level':0, 'print_time':0, 'ipopt.acceptable_tol':1e-8, 'ipopt.acceptable_obj_change_tol':1e-6}
    opti.solver('ipopt',opts_setting)
    opti.set_value(opt_x0, attacker_state)
    sol = opti.solve()
    u_res = sol.value(opt_controls)

    return u_res[0, :]

def Base_NMPC(attacker_state,defender_state):

    opti = ca.Opti()

    ## parameters
    T = 0.05
    N = 10
    v_max = 0.5
    omega_max = 2
    Q = np.array([[1.0, 0.0, 0.0],[0.0, 1.0, 0.0],[0.0, 0.0, 0.0]])
    R = np.array([[0.5, 0.0], [0.0, 0.001]])
    W_slack = np.array([[1000]])
    goal = np.array([[1.2, 1.2, 0.0]])

    ## control variables, linear velocity v and angle velocity omega
    opt_x0 = opti.parameter(3)
    opt_controls = opti.variable(N, 2)
    v = opt_controls[:, 0]
    omega = opt_controls[:, 1]

    ## state variables
    opt_states = opti.variable(N+1, 3)
    x = opt_states[:, 0]
    y = opt_states[:, 1]
    theta = opt_states[:, 2]

    ## slack variable to avoid infeasibility 
    d_slack = opti.variable(N, 1)

    ## create funciont for F(x), H(x) and V(x)
    f = lambda x_, u_: ca.vertcat(*[u_[0]*ca.cos(x_[2]), u_[0]*ca.sin(x_[2]), u_[1]])

    ## init_condition
    opti.subject_to(opt_states[0, :] == opt_x0.T)
    opti.subject_to(opti.bounded(-1.45, x, 1.45))
    opti.subject_to(opti.bounded(-1.45, y, 1.45))
    opti.subject_to(opti.bounded(-v_max, v, v_max))
    opti.subject_to(opti.bounded(-omega_max, omega, omega_max))   

    ## system model constrain
    for i in range(N):
        x_next = opt_states[i, :] + f(opt_states[i, :], opt_controls[i, :]).T*T
        opti.subject_to(opt_states[i+1, :]==x_next)
    
    ## collision avoid constrain
    for i in range(N):
        distance_constraints = (opt_states[i, 0].T - defender_state[0]) ** 2 + (opt_states[i, 1].T - defender_state[1]) ** 2 
        opti.subject_to(distance_constraints >= 0.16 + d_slack[i, :])

    #### cost function
    obj = 0 
    for i in range(N):
        obj = obj + ca.mtimes([(opt_states[i, :] - goal), Q, (opt_states[i, :]- goal).T]) + ca.mtimes([opt_controls[i, :], R, opt_controls[i, :].T]) + ca.mtimes([d_slack[i, :], W_slack, d_slack[i, :].T]) 

    opti.minimize(obj)
    opts_setting = {'ipopt.max_iter':200, 'ipopt.print_level':0, 'print_time':0, 'ipopt.acceptable_tol':1e-8, 'ipopt.acceptable_obj_change_tol':1e-6}
    opti.solver('ipopt',opts_setting)
    opti.set_value(opt_x0, attacker_state)
    sol = opti.solve()
    u_res = sol.value(opt_controls)

    return u_res[0, :]

def Simple_Catcher(attacker_state,defender_state):
    is_poistive = 1
    distance = np.sqrt(np.sum(np.square(attacker_state[:2] - defender_state[:2])))
    dx =attacker_state[0] - defender_state[0]
    dy =attacker_state[1] - defender_state[1]
    theta_e = np.arctan2(dy,dx) - defender_state[2]
    # print(np.arctan(dy/dx))
    # print(defender_state[2])
    # attacker_state[2] - defender_state[2]
    if(theta_e>np.pi):
        theta_e -= 2*np.pi
    elif (theta_e<-np.pi):
        theta_e += 2*np.pi
    # print(theta_e)
    
    if(theta_e>np.pi/2 or theta_e<-np.pi/2):
        is_poistive = -1

    u = 1.5*distance*is_poistive
    w = theta_e*is_poistive
    u = np.clip(u, -0.5, 0.5)
    w = np.clip(w, -2, 2)
    return np.array([u,w])




        



if __name__ == '__main__':
    states = np.array([[-1,0,0],[0.5,0,-np.pi]])
    u = CLF_CBF_NMPC(states[0],states[1])
    print(u)