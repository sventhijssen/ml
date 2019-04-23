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

    def get_mesh_dynamics(self, xs_mesh, ys_mesh, player):
        """
        Returns the dynamics for the given populations of the different players.
        xs represents the population of the first player. ys represents the population of the second player.
        :param xs_mesh: An array of arrays. Each array contains the x1-values. The complementary population x2 = 1-x1.
        :param ys_mesh: An array of arrays. Each array contains the y1-values. The complementary population y2 = 1-y1.
        :param player: Integer. Indicates the player.
        :return:
        """
        return [self.get_dynamics(xs_mesh[i], ys_mesh[i], player) for i in range(len(xs_mesh))]