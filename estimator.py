import numpy as np

class Estimator:
    """
    A simple estimator that predicts systems' future states

    Given systems' dynamics, inital states, time step, and a prediciton horizon, this estimator returns all future states of the agent governed by given dynamics within the prediction horizon. 

    Attributes: 
        cur_state       : current state
        ctrl_input      : control input u of dynamics f(x, u)
        step            : discretization time step
        dynamics        : system evolving dynamics f(x, u)
        window          : prediciton horizon
        predict_states  : predicting results
    """
    cur_state = np.array([0])
    ctrl_input = np.array([0])
    step = 0.1
    dynamics = lambda x_, v_: x_ + step * v_
    window = 1
    predict_states = np.zeros(window)

    def __init__(self, x0, u, dyn, stp, windw):
        self.cur_state = x0
        self.ctrl_input = u
        self.step = stp
        self.dynamics = dyn
        self.window = windw
        self.predict_states = np.zeros(windw)

    def predict(self):
        x0 = self.cur_state
        dyn = self.dynamics
        u = self.ctrl_input

        # System evolves: x(t+1) = x(t) + T * f(x(t), u) 
        x_nxt = x0 + self.step * dyn(x0, u)
        self.predict_states = np.array([x_nxt])
        x0 = x_nxt
        for t in range(1, self.window):
            x_nxt = x0 + self.step * dyn(x0, u)
            self.predict_states = np.concatenate((self.predict_states, np.array([x_nxt])), axis=0)
            x0 = x_nxt



## Test code
if __name__ == '__main__':
    # Unicycle model
    dyn = lambda x_, u_: np.array([u_.T[0]*np.cos(x_.T[2]), u_.T[0]*np.sin(x_.T[2]), u_.T[1]]).T
    est = Estimator(np.array([[0, 0, 0], [1,2,3]]), np.array([[1, 0], [-1, 0]]), dyn, 0.1, 10)
    est.predict()
    print(est.predict_states)


