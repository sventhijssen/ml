from matplotlib.pyplot import show, subplots, axis, contour
from numpy import meshgrid, array, arange

from Dynamics import Dynamics
from MatchingPenniesEnvironment import MatchingPenniesEnvironment
from PrisonersDilemmaEnvironment import PrisonersDilemmaEnvironment
from Player import Player


def independent_learning(environment):
    nr_episodes = 100

    player_one = Player()
    player_two = Player()

    for k in range(nr_episodes):
        environment.set_action_player_one(player_one.get_action(k))
        environment.set_action_player_two(player_two.get_action(k))

        player_one.update_q_table(environment.get_action_player_one(), environment.get_reward_player_one())
        player_two.update_q_table(environment.get_action_player_two(), environment.get_reward_player_two())

    print(player_one.q_table)
    print(player_two.q_table)


def dynamics_learning(environment):
    # DYNAMICS
    dynamics = Dynamics(environment)

    # https://stackoverflow.com/questions/1843194/plotting-vector-fields-in-python-matplotlib

    # arange(start, stop, step)
    xs_mesh, ys_mesh = meshgrid(arange(0, 1.1, .1), arange(0, 1.1, .1))

    us = array(dynamics.get_mesh_dynamics(xs_mesh, ys_mesh, 0))
    vs = array(dynamics.get_mesh_dynamics(ys_mesh, xs_mesh, 1))
    zs_mesh = ys_mesh - xs_mesh

    fig, ax = subplots()
    ax.quiver(xs_mesh, ys_mesh, us, vs)
    contour(xs_mesh, ys_mesh, zs_mesh, [0.5, 1.0, 1.2, 1.5], colors='k', linestyles = 'solid')
    show()


def main():
    print("Matching Pennies Environment")
    mpe = MatchingPenniesEnvironment()
    independent_learning(mpe)
    # dynamics_learning(mpe)

    print("Prisoner's Dilemma Environment")
    pde = PrisonersDilemmaEnvironment()
    independent_learning(pde)
    # dynamics_learning(pde)


if __name__ == "__main__":
    main()
