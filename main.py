from matplotlib.pyplot import show, subplots
from numpy import meshgrid, array, arange

from Dynamics import Dynamics
from MatchingPenniesEnvironment import MatchingPenniesEnvironment
from QLearning import QLearning
from PrisonersDilemmaEnvironment import PrisonersDilemmaEnvironment

def main():
    environment = MatchingPenniesEnvironment()
    # nr_episodes = 10000
    #
    # player_a = QLearning(environment)
    # player_b = QLearning(environment)
    # for ep in range(nr_episodes):
    #     environment.action_player_one(player_a.action())
    #     environment.action_player_two(player_b.action())
    #
    #     player_a.reward(environment.PlayerOneAction, environment.reward_player_one())
    #     player_b.reward(environment.PlayerTwoAction, environment.reward_player_two())
    #
    #     print(player_a.q_table)
    #     print(player_b.q_table)


    # DYNAMICS
    dynamics = Dynamics(environment)

    # https://stackoverflow.com/questions/1843194/plotting-vector-fields-in-python-matplotlib

    # arange(start, stop, step)
    xs_mesh, ys_mesh = meshgrid(arange(0, 1.1, .1), arange(0, 1.1, .1))

    us = array(dynamics.get_mesh_dynamics(xs_mesh, ys_mesh, 0))
    vs = array(dynamics.get_mesh_dynamics(ys_mesh, xs_mesh, 1))

    fig, ax = subplots()
    ax.quiver(xs_mesh, ys_mesh, us, vs)
    show()

if __name__ == "__main__":
    main()
