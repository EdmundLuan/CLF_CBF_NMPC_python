import numpy as np
import casadi as ca
from estimator import Estimator
from observer import Observer



def CLF_CBF_NMPC(self_state, others_states, estmtr):
    
    opti = ca.Opti()

    ## parameters for optimization
    T = 0.1
    N = 14  # MPC horizon
    M_CBF = 3  # CBF-QP horizon
    M_CLF = 2   # CLF-QP horizon
    gamma_k = 0.1
    v_max = 0.5
    omega_max = 1.5
    safe_distance = 0.24
    Q = np.array([[1.0, 0.0, 0.0],[0.0, 1.0, 0.0],[0.0, 0.0, 0.0]])
    R = np.array([[0.1, 0.0], [0.0, 0.0001]])
    # R = np.array([[0.8, 0.0], [0.0, 0.001]])
    W_slack = np.array([[1000]])
    goal = np.array([[1.5, 1.5, 0.0]])

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
    # print(np.size(others_states, 0))
    for idx in range(np.size(others_states, 0)):
        # CBF
        h = lambda x_,y_: (x_[0] - y_[0]) ** 2 + (x_[1] - y_[1]) ** 2 - safe_distance**2
    # CLF
    V = lambda x_: (x_[0] - goal[0][0]) ** 2 + (x_[1] - goal[0][1]) ** 2 

    ## init_condition
    opti.subject_to(opt_states[0, :] == opt_x0.T)

    # Position Boundaries
    opti.subject_to(opti.bounded(-1.45, x, 1.45))
    opti.subject_to(opti.bounded(-1.45, y, 1.45))
    # Admissable Control constraints
    opti.subject_to(opti.bounded(-v_max, v, v_max))
    opti.subject_to(opti.bounded(-omega_max, omega, omega_max)) 

    # System Model constraints
    for i in range(N):
        x_next = opt_states[i, :] + T*f(opt_states[i, :], opt_controls[i, :]).T
        opti.subject_to(opt_states[i+1, :]==x_next)
    
    # CBF constraints
    estmtr.window = M_CBF
    for i in range(M_CBF):
        for j in range(np.size(others_states, 0)):
            opti.subject_to(h(opt_states[i+1, :], others_states[j]) >= (1-gamma_k)*h(opt_states[i, :],others_states[j]) ) 
        for j in range(np.size(estmtr.predict_states, 0)):
            opti.subject_to( h(opt_states[i+1, :], estmtr.predict_states[j][i]) >= (1-gamma_k)*h(opt_states[i, :], estmtr.predict_states[j][i]) )
            # print("Predicted state of agent %d at %d steps later:" % (j+1, i+1) )
            # print(estmtr.predict_states[j][i])
        

    # # CLF constraints
    # for i in range(M_CLF):
    #     opti.subject_to(V(opt_states[i+1, :]) <= (1-0.1)*V(opt_states[i, :]) + d_slack[i, :]) 

    #### cost function
    obj = 0 
    for i in range(N):
        obj = obj + ca.mtimes([opt_controls[i, :], R, opt_controls[i, :].T]) + ca.mtimes([d_slack[i, :], W_slack, d_slack[i, :].T]) 
    obj = obj + ca.mtimes([(opt_states[N-1, :] - goal), Q, (opt_states[N-1, :]- goal).T])

    opti.minimize(obj)
    opts_setting = {'ipopt.max_iter':500, 'ipopt.print_level':0, 'print_time':0, 'ipopt.acceptable_tol':1e-5, 'ipopt.acceptable_obj_change_tol':1e-6}
    opti.solver('ipopt',opts_setting)
    opti.set_value(opt_x0, self_state)
    sol = opti.solve()
    u_res = sol.value(opt_controls)

    return u_res[0, :]

def Base_NMPC(self_state, others_states):
    opti = ca.Opti()

    ## parameters for optimization
    T = 0.1
    N = 10  # MPC horizon
    M_CBF = 6  # CBF-QP horizon
    M_CLF = 2   # CLF-QP horizon
    gamma_k = 0.1
    v_max = 0.5
    omega_max = 1.5
    safe_distance = 0.4
    Q = np.array([[2.0, 0.0, 0.0],[0.0, 2.0, 0.0],[0.0, 0.0, 0.0]])
    R = np.array([[0.1, 0.0], [0.0, 0.0001]])
    # R = np.array([[0.8, 0.0], [0.0, 0.001]])
    W_slack = np.array([[800]])
    goal = np.array([[1.35, 1.35, 0.0]])

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
    h = []
    # print(np.size(others_states, 0))
    # for idx in range(np.size(others_states, 0)):
    #     h.append( lambda x_: (x_[0] - others_states[idx][0]) ** 2 + (x_[1] - others_states[idx][1]) ** 2 - safe_distance**2 )
    h1 = lambda x_: (x_[0] - others_states[0][0]) ** 2 + (x_[1] - others_states[0][1]) ** 2 - safe_distance**2
    h2 = lambda x_: (x_[0] - others_states[1][0]) ** 2 + (x_[1] - others_states[1][1]) ** 2 - safe_distance**2
    h3 = lambda x_: (x_[0] - others_states[2][0]) ** 2 + (x_[1] - others_states[2][1]) ** 2 - safe_distance**2
    h4 = lambda x_: (x_[0] - others_states[3][0]) ** 2 + (x_[1] - others_states[3][1]) ** 2 - safe_distance**2
    h5 = lambda x_: (x_[0] - others_states[4][0]) ** 2 + (x_[1] - others_states[4][1]) ** 2 - safe_distance**2
    # h6 = lambda x_: (x_[0] - others_states[5][0]) ** 2 + (x_[1] - others_states[5][1]) ** 2 - safe_distance**2
    
    V = lambda x_: (x_[0] - goal[0][0]) ** 2 + (x_[1] - goal[0][1]) ** 2 

    ## init_condition
    opti.subject_to(opt_states[0, :] == opt_x0.T)

    # Position Boundaries
    opti.subject_to(opti.bounded(-1.45, x, 1.45))
    opti.subject_to(opti.bounded(-1.45, y, 1.45))
    # Admissable Control constraints
    opti.subject_to(opti.bounded(-v_max, v, v_max))
    opti.subject_to(opti.bounded(-omega_max, omega, omega_max)) 

    # system model constraints
    for i in range(N):
        x_next = opt_states[i, :] + f(opt_states[i, :], opt_controls[i, :]).T*T
        opti.subject_to(opt_states[i+1, :]==x_next)
    
    # CBF constraints
    for i in range(N):
        # for j in range(np.size(others_states, 0)):
        #     opti.subject_to(h[j](opt_states[i+1, :]) >= (1-gamma_k)*h[j](opt_states[i, :]) ) 
        opti.subject_to(h1(opt_states[i, :]) > 0 )
        opti.subject_to(h2(opt_states[i, :]) > 0 )
        opti.subject_to(h3(opt_states[i, :]) > 0)
        opti.subject_to(h4(opt_states[i, :]) > 0 )
        opti.subject_to(h5(opt_states[i, :]) > 0 )

    # CLF constraints
    # for i in range(M_CLF):
    #     opti.subject_to(V(opt_states[i+1, :]) <= (1-0.1)*V(opt_states[i, :]) + d_slack[i, :]) 

    #### cost function
    obj = 0 
    for i in range(N):
        obj = obj + ca.mtimes([opt_controls[i, :], R, opt_controls[i, :].T]) + ca.mtimes([d_slack[i, :], W_slack, d_slack[i, :].T]) 
    obj = obj + ca.mtimes([(opt_states[N-1, :] - goal), Q, (opt_states[N-1, :]- goal).T])

    opti.minimize(obj)
    opts_setting = {'ipopt.max_iter':200, 'ipopt.print_level':0, 'print_time':0, 'ipopt.acceptable_tol':1e-5, 'ipopt.acceptable_obj_change_tol':1e-5}
    opti.solver('ipopt',opts_setting)
    opti.set_value(opt_x0, self_state)
    sol = opti.solve()
    u_res = sol.value(opt_controls)

    return u_res[0, :]

    # opti = ca.Opti()

    # ## parameters
    # T = 0.05
    # N = 10
    # v_max = 0.5
    # omega_max = 2
    # Q = np.array([[1.0, 0.0, 0.0],[0.0, 1.0, 0.0],[0.0, 0.0, 0.0]])
    # R = np.array([[0.5, 0.0], [0.0, 0.001]])
    # W_slack = np.array([[1000]])
    # goal = np.array([[1.2, 1.2, 0.0]])

    # ## control variables, linear velocity v and angle velocity omega
    # opt_x0 = opti.parameter(3)
    # opt_controls = opti.variable(N, 2)
    # v = opt_controls[:, 0]
    # omega = opt_controls[:, 1]

    # ## state variables
    # opt_states = opti.variable(N+1, 3)
    # x = opt_states[:, 0]
    # y = opt_states[:, 1]
    # theta = opt_states[:, 2]

    # ## slack variable to avoid infeasibility 
    # d_slack = opti.variable(N, 1)

    # ## create funciont for F(x), H(x) and V(x)
    # f = lambda x_, u_: ca.vertcat(*[u_[0]*ca.cos(x_[2]), u_[0]*ca.sin(x_[2]), u_[1]])

    # ## init_condition
    # opti.subject_to(opt_states[0, :] == opt_x0.T)
    # opti.subject_to(opti.bounded(-1.45, x, 1.45))
    # opti.subject_to(opti.bounded(-1.45, y, 1.45))
    # opti.subject_to(opti.bounded(-v_max, v, v_max))
    # opti.subject_to(opti.bounded(-omega_max, omega, omega_max))   

    # ## system model constrain
    # for i in range(N):
    #     x_next = opt_states[i, :] + f(opt_states[i, :], opt_controls[i, :]).T*T
    #     opti.subject_to(opt_states[i+1, :]==x_next)
    
    # ## collision avoid constrain
    # for i in range(N):
    #     distance_constraints = (opt_states[i, 0].T - defender_state[0]) ** 2 + (opt_states[i, 1].T - defender_state[1]) ** 2 
    #     opti.subject_to(distance_constraints >= 0.16 + d_slack[i, :])

    # #### cost function
    # obj = 0 
    # for i in range(N):
    #     obj = obj + ca.mtimes([(opt_states[i, :] - goal), Q, (opt_states[i, :]- goal).T]) + ca.mtimes([opt_controls[i, :], R, opt_controls[i, :].T]) + ca.mtimes([d_slack[i, :], W_slack, d_slack[i, :].T]) 

    # opti.minimize(obj)
    # opts_setting = {'ipopt.max_iter':500, 'ipopt.print_level':0, 'print_time':0, 'ipopt.acceptable_tol':1e-6, 'ipopt.acceptable_obj_change_tol':1e-6}
    # opti.solver('ipopt',opts_setting)
    # opti.set_value(opt_x0, attacker_state)
    # sol = opti.solve()
    # u_res = sol.value(opt_controls)

    # return u_res[0, :]

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
    u = np.clip(u, -0.32, 0.32)
    w = np.clip(w, -2, 2)
    return np.array([u,w])




        



if __name__ == '__main__':
    states = np.array([[-1,0,0],[0.5,0,-np.pi]])
    u = CLF_CBF_NMPC(states[0],states[1])
    print(u)