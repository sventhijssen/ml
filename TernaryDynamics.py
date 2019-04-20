import numpy as np


class TernaryDynamics:

    def __init__(self, environment):
        self.environment = environment

    def get_change(self, x, y, player):
        # print("(" + str(x) + ", " + str(y) + ")")
        if player == 0:
            x_vec = np.array(x)
            y_vec = np.array(y)
            payoff_matrix = np.array(self.environment.get_payoff_matrix(player))
            a = np.dot(x_vec, payoff_matrix)
            c = np.dot(a, np.transpose(y))
            r = x_vec
            for i in range(len(x_vec)):
                b = np.dot(payoff_matrix, np.transpose(y_vec))[i]
                r[i] = x_vec[i]*(b - c)
        else:
            x_vec = np.array(x)
            y_vec = np.array(y)
            payoff_matrix = np.array(self.environment.get_payoff_matrix(player))
            a = np.dot(x_vec, payoff_matrix)
            c = np.dot(a, np.transpose(y))
            r = y_vec
            for i in range(len(y_vec)):
                b = np.dot(x_vec, payoff_matrix)[i]
                r[i] = y_vec[i]*(b - c)
        return np.array(r)

    def get_dynamics(self, combs):
        return [self.get_change(combs[i][0], combs[i][1]) for i in range(len(combs))]
