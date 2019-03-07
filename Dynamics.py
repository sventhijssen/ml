import numpy as np


class Dynamics:

    def __init__(self, environment):
        self.environment = environment

    def get_change(self, x, y, player):
        x_vec = np.array([x, 1-x])
        y_vec = np.array([y, 1-y])
        payoff_matrix = np.array(self.environment.get_payoff_matrix(player))
        a = np.dot(x_vec, payoff_matrix)
        c = np.dot(a, np.transpose(y_vec))
        b = payoff_matrix[0][0] * y_vec[0] + payoff_matrix[0][1] * y_vec[1]
        return x_vec[0]*(b - c)

    def get_dynamics(self, xs, ys, player):
        return [self.get_change(x, y, player) for x in xs for y in ys]

    def get_mesh_dynamics(self, xs_mesh, ys_mesh, player):
        """
        Returns the dynamics for the given populations of the different players.
        xs represents the population of the first player. ys represents the population of the second player.
        :param xs_mesh: An array of arrays. Each array contains the x1-values. The complementary population x2 = 1-x1.
        :param ys_mesh: An array of arrays. Each array contains the y1-values. The complementary population y2 = 1-y1.
        :param player: Integer. Indicates the player.
        :return:
        """
        return [self.get_dynamics(xs, ys, player) for xs in xs_mesh for ys in ys_mesh]
