import numpy as np


class TernaryDynamics:

    def __init__(self, environment):
        self.environment = environment

    def get_change(self, x, y):
        # print("(" + str(x) + ", " + str(y) + ")")
        x_vec = np.array(x)
        y_vec = np.array(y)
        payoff_matrix = np.array(self.environment.get_payoff_matrix(0))
        a = np.dot(x_vec, payoff_matrix)
        c = np.dot(a, np.transpose(y))
        result_x = x_vec
        for i in range(len(x_vec)):
            b = np.dot(payoff_matrix, np.transpose(y_vec))[i]
            result_x[i] = x_vec[i]*(b - c)
        return np.array(result_x)

    def get_dynamics(self, combs):
        return [self.get_change(combs[i][0], combs[i][1]) for i in range(len(combs))]
