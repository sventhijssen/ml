import numpy as np


class Dynamics:

    def __init__(self, environment):
        self.environment = environment

    def get_change(self, x, y, player):
        if player == 0:
            x_vec = np.array([x, 1-x])
            y_vec = np.array([y, 1-y])
            payoff_matrix = np.array(self.environment.get_payoff_matrix(player))
            a = np.dot(x_vec, payoff_matrix)
            c = np.dot(a, np.transpose(y_vec))
            b = np.dot(payoff_matrix, np.transpose(y_vec))[0]
            r = x_vec[0]*(b - c)
        else:
            x_vec = np.array([x, 1-x])
            y_vec = np.array([y, 1-y])
            payoff_matrix = np.array(self.environment.get_payoff_matrix(player))
            a = np.dot(x_vec, payoff_matrix)
            c = np.dot(a, np.transpose(y_vec))
            b = x_vec[0] * payoff_matrix[0][0] + x_vec[1] * payoff_matrix[1][0]
            r = y_vec[0]*(b - c)
        return r

    def get_dynamics(self, xs, ys, player):
        return [self.get_change(xs[i], ys[i], player) for i in range(len(ys))]

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
