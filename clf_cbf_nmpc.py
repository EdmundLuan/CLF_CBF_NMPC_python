# MIT License

# Copyright (c) 2021 Hao Luan, Anxing Xiao

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import numpy as np
import casadi as ca

class CLF_CBF_NMPC():
    """
    A class encapsulating the CLF-CBF-NMPC method, implemented using CasADi. 

    Detailed description. 

    Attributes:
        N               : MPC horizon
        M_CBF           : CBF constraints horizon
        M_CLF           : CLF constraints horizon
        gamma_k         : (1-\\gamma_k) is the exponential decreasing rate of a CBF at time step k
        alpha_k         : (1-\\alpha_k) is the exponential decreasing rate of a CLF at time step k
        step            : discretization time step in seconds 
        goal_global     : goal state in global coordinates
        
        opti            : CasADi's Opti stack
        opt_x0_global   : initial state in global coordinates
        opt_x0_local   : initial state in global coordinates
        opt_states      : optimized states/trajectory
        opt_controls    : optimized control variables
        opt_trj         : optimized trajectory (in global coordinates)
        opt_d_slack     : a slack variable resolving infeasibility between CLF and CBF
        opt_cost        : 


        goal_local      : goal state in local coordinates
        v_max           : Maximum linear velocity
        omega_max       : Maximum angular velocity
    """

    ## parameters for optimization
    step = 0.1
    N = 12 
    M_CBF = 6 
    M_CLF = 1 
    gamma_k = 0.1 
    alpha_k = 0.01 
    opti = ca.Opti()

    opt_x0_global = opti.parameter()
    opt_x0_local = opti.parameter()
    opt_controls = opti.variable(N)
    opt_states = opti.variable(N+1)
    opt_d_slack = opti.variable(N)
    opt_cost = 0

    v_max = 0.5
    omega_max = 1.5

    def __init__(self, n, stp, m_cbf, m_clf, gamma, alpha, dim_x = 3, dim_u = 2):
        self.N = n
        self.step = stp
        self.M_CBF = m_cbf
        self.M_CLF = m_clf
        self.gamma_k = gamma
        self.alpha_k = alpha
        self.opti = ca.Opti()

        self.v_max = 0.45
        self.omega_max = 1.5
        # self.v_max = v_m
        # self.omega_max = omg_m

        # state variables
        self.opt_states = self.opti.variable(self.N+1, dim_x)
        # control variables: linear velocity v and angle velocity omega
        self.opt_controls = self.opti.variable(self.N, dim_u)
        # a slack variable to avoid infeasibility between CLF and CBF constraints
        self.opt_d_slack = self.opti.variable(self.N, 1)
        
        # initial state
        self.opt_x0_local = self.opti.parameter(dim_x)
        self.opt_x0_global = self.opti.parameter(dim_x)
        self.goal_global = self.opti.parameter(dim_x)
        self.goal_local = self.opti.parameter(dim_x)
        
        # set up cost function
        self.__set_cost_func()


    def reset(self):
        # clear all constraints
        self.opti.subject_to()
        self.opt_cost = 0
        self.__set_cost_func()

    def set_goal(self, goal_state):
        """
        Set goal states (in globla coordinates)

        Turn a vector into a matrix. 

        Args:
            goal_global:        : vector/list
        
        Returns:
            None.
        """
        self.opti.set_value(self.goal_global, goal_state)
        self.opti.set_value(self.goal_local, goal_state)

    def __set_cost_func(self):
        # quadratic cost  : J = (x-goal)^T * Q * (x-goal)  +  u^T * R * u  + W_slack * opt_d_slack 
        # Q               : Weights for state variables in a (quadratic) cost function 
        # R               : Weights for control input variables in a (quadratic) cost function
        Q = np.array([[1, 0.0, 0.0],[0.0, 1, 0.0],[0.0, 0.0, 0.0]])
        R = np.array([[0.01, 0.0], [0.0, 0.00001]])
        W_slack = np.array([[1000]])
        for i in range(self.N):
            self.opt_cost = self.opt_cost + ca.mtimes([self.opt_controls[i, :], R, self.opt_controls[i, :].T]) + ca.mtimes([self.opt_d_slack[i, :], W_slack, self.opt_d_slack[i, :].T])
            # self.opt_cost = self.opt_cost + ca.mtimes([(self.opt_states[i, :] - self.goal_local.T), Q, (self.opt_states[i, :]- self.goal_local.T).T]) / 5
        # final term 
        self.opt_cost = self.opt_cost + ca.mtimes([(self.opt_states[self.N-1, :] - self.goal_local.T), Q, (self.opt_states[self.N-1, :]- self.goal_local.T).T])


    def __global2local(self):
        """
        Transform global coordinates into local coordinates w.r.t. the robot

        Robot is always at the origin (0,0, theta). Orientation is left as is. 

        Args: 
            None.
        Returns:
            None.
        """
        init_st = self.opti.value(self.opt_x0_global)
        # self.opti.set_value(self.goal_local, self.opti.value(self.goal_global) - init_st)
        # self.opti.set_value(self.opt_x0_local, self.opti.value(self.opt_x0_global) - init_st)
        self.goal_local[0] = self.goal_global[0] -init_st[0]
        self.goal_local[1] = self.goal_global[1] -init_st[1]
        self.opt_x0_local[0] = self.opt_x0_global[0] -init_st[0]
        self.opt_x0_local[1] = self.opt_x0_global[1] -init_st[1]
        # time k
        for t_k in self.obstacles:
            # obstacle i at time k
            for obs_i in t_k:
                obs_i[0] -= init_st[0]
                obs_i[1] -= init_st[1]
                

        # print('Goal_global:')
        # print(self.opti.value(self.goal_global))
        # # print('Goal_local:')
        # print(self.opti.value(self.goal_local))

    def __local2global(self):
        delta = self.opti.value(self.opt_x0_global-self.opt_x0_local)
        self.opt_trj = self.solution.value(self.opt_states).copy()
        # state at time k
        for st_k in self.opt_trj:
            # only transform back x and y coordinates
            self.opt_trj[0] += delta[0]
            self.opt_trj[1] += delta[1]
    
    def __add_system_constrnts(self): 
        x = self.opt_states[:, 0]
        y = self.opt_states[:, 1]
        # theta = self.opt_states[:, 2]
        v = self.opt_controls[:, 0]
        omega = self.opt_controls[:, 1]
        # position boundaries (not necessary)
        delta = self.opti.value(self.opt_x0_global-self.opt_x0_local)
        self.opti.subject_to(self.opti.bounded(-1.45, x+delta[0], 1.45))
        self.opti.subject_to(self.opti.bounded(-1.45, y+delta[1], 1.45))
        # admissable control constraints
        self.opti.subject_to(self.opti.bounded(-self.v_max, v, self.v_max))
        self.opti.subject_to(self.opti.bounded(-self.omega_max, omega, self.omega_max)) 

    def __add_dynamics_constrnts(self):
        # create system dynamics x(t+1) = x(t) + f(x(t), u) * step
        f = lambda x_, u_: ca.vertcat(*[u_[0]*ca.cos(x_[2]), u_[0]*ca.sin(x_[2]), u_[1]])
        
        # initial condition constraint
        self.opti.subject_to(self.opt_states[0, :] == self.opt_x0_local.T)  # In local, x0 == [vec(0), theta]
        # self.opti.subject_to(self.opt_states[0, :] == self.opt_x0.T)

        # Dynamics constraints
        for i in range(self.N):
            x_next = self.opt_states[i, :] + self.step*f(self.opt_states[i, :], self.opt_controls[i, :]).T
            self.opti.subject_to(self.opt_states[i+1, :]==x_next)

    def __add_safe_constrnts(self):
        # safe distance
        safe_dist = 0.24
        # control barrier function
        h = lambda x_,y_: (x_[0] - y_[0]) ** 2 + (x_[1] - y_[1]) ** 2 - safe_dist**2
        
        # add CBF constraints
        for i in range(self.M_CBF):
            # current and future safety contraints
            # others_states[i][j]: agent No. j+1 's state at i steps ahead, these are provided by an observer & a predictor
            for j in range(np.size(self.obstacles, 1)):
                self.opti.subject_to(h(self.opt_states[i+1, :], self.obstacles[i+1][j]) >= (1-self.gamma_k)*h(self.opt_states[i, :], self.obstacles[i][j]) ) 

    def __add_stability_constrnts(self):
        # control Lyapunov function 
        V = lambda x_: (x_[0] - self.goal_local[0]) ** 2 + (x_[1] - self.goal_local[1]) ** 2 
        # add CLF constraints
        for i in range(self.M_CLF):
            self.opti.subject_to(V(self.opt_states[i+1, :]) <= (1-self.alpha_k)*V(self.opt_states[i, :]) + self.opt_d_slack[i, :])
        pass

    def __adopt_d_slack_constrnts(self):
        pass

    def solve(self, init_st, goal_st, others_states):
        """
        Solving the CLF-CBF-NMPC optimization. 

        Initializing the solver is required before first-time using. 


        Args: 
            init_st             : initial state of the robot
            others_states       : all other agents' (or obstacles') states at CURRENT step AND some FUTURE steps 

        Returns:
            global_states_sol   : optimized trajectory (in global coordinates)
            controls_sol        : optimized control inputs trajectory 
            d_slck_sol          : optimized slack variable
            local_states_sol    : optimized trajectory (in local coordinates)
        """
        # Reset the optimizer FIRST!
        self.reset()
        # print("Optimizer reset.")
        
        # set initial states
        self.opti.set_value(self.opt_x0_global, init_st)
        self.opti.set_value(self.opt_x0_local, init_st)
        # set goal states
        self.set_goal(goal_st)

        # obstacles or other agents
        self.obstacles = others_states.copy()
        # only care about obstacles within an ROI 
        l = []
        for i in range(self.obstacles.shape[1]):
            if((self.obstacles[0][i][0] - init_st[0])** 2 + (self.obstacles[0][i][1] - init_st[1])** 2 > 1):
                l.append(i)
        # delete those out of the ROI
        self.obstacles = np.delete(self.obstacles, l, axis=1)
        # print(self.obstacles.shape)


        # Turn everything into local coordinates. 
        self.__global2local()       # Very IMPORTANT!! 
        # print('Coordinates transformed to local.')

        # Adding constraints to the optimizer
        self.__add_system_constrnts()
        self.__add_dynamics_constrnts()
        self.__add_safe_constrnts()
        self.__add_stability_constrnts()

        # Optimizer configures
        self.opti.minimize(self.opt_cost)
        opts_setting = {'ipopt.max_iter':200, 'ipopt.print_level':1, 'print_time':0, 'ipopt.acceptable_tol':1e-5, 'ipopt.acceptable_obj_change_tol':1e-5}
        self.opti.solver('ipopt', opts_setting)
        # self.opti.solver('ipopt')
        # self.solution = self.opti.solve()
        

        # Solve
        try:
            self.solution = self.opti.solve()
            local_states_sol = self.opti.value(self.opt_states)
            controls_sol = self.opti.value(self.opt_controls)
            # d_slck_sol = self.opti.value(self.opt_d_slack)
            self.__local2global()
            global_states_sol = self.opt_trj
        except:
            print('Something went wrong!!')
            local_states_sol = self.opti.debug.value(self.opt_states)
            controls_sol = self.opti.debug.value(self.opt_controls)
            # d_slck_sol = self.opti.debug.value(self.opt_d_slack)
            self.__local2global()
            global_states_sol = self.opt_trj

        return global_states_sol, controls_sol, local_states_sol #, d_slck_sol,
        # return self.solution
    
    def get_optimized_u(self):
        print('Optimized controls:')
        print(self.solution.value(self.opt_controls)[0, :])
        return self.solution.value(self.opt_controls)[0, :]



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


## Test Code
if __name__ == '__main__':
    states = np.array([[-1,0,0],[0.5,0,-np.pi]])
    u = CLF_CBF_NMPC(states[0],states[1])
    print(u)
