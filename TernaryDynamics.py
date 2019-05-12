import numpy as np
"""
Calculate dynamics for ternary games.
"""
class TernaryDynamics:

    def __init__(self, environment):
        self.environment = environment

    def get_change(self, x):
        x_vec = np.array(x)
        payoff_matrix = np.array(self.environment.get_payoff_matrix(0))
        print("payoff")
        print(payoff_matrix)
        f_s = np.dot(payoff_matrix, x_vec)
        print("fs")
        print(f_s)
        f_avg = np.ones(3)*np.dot(x_vec, f_s)
        f_avg2 = x_vec[0]*f_s[0]+x_vec[1]*f_s[1]+x_vec[2]*f_s[2]
        print("avg")
        print(f_avg)
        print(f_avg2)
        r = np.array(x_vec*(f_s - f_avg2))
        return r
