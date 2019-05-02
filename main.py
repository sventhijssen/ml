from matplotlib.pyplot import show, subplots, axis, savefig, figure
from numpy import meshgrid, array, arange, zeros, matrix

from DQNPlayer_Keras import DQNPlayer
from Dynamics import Dynamics
from MatchingPenniesEnvironment import MatchingPenniesEnvironment
from PrisonersDilemmaEnvironment import PrisonersDilemmaEnvironment
from Player import Player
import pylab as p


def independent_learning(environment, player_one, player_two):
    nr_episodes = 100000

    prob_one = []
    prob_two = []

    for k in range(nr_episodes):
        environment.set_action_player_one(player_one.get_action(k))
        environment.set_action_player_two(player_two.get_action(k))

        player_one.update_q_table(environment.get_action_player_one(), environment.get_reward_player_one())
        player_two.update_q_table(environment.get_action_player_two(), environment.get_reward_player_two())

        prob_one.append(player_one.get_probability_action(0, k))
        prob_two.append(player_two.get_probability_action(0, k))

    return player_one.get_q_table(), player_two.get_q_table(), prob_one, prob_two


def dynamics_learning(environment):
    # DYNAMICS
    dynamics = Dynamics(environment)

    # https://stackoverflow.com/questions/1843194/plotting-vector-fields-in-python-matplotlib

    # arange(start, stop, step)
    xs_mesh, ys_mesh = meshgrid(arange(0, 1.05, .1), arange(0, 1.05, .1))

    us = array(dynamics.get_mesh_dynamics(xs_mesh, ys_mesh, 0))
    vs = array(dynamics.get_mesh_dynamics(xs_mesh, ys_mesh, 1))
    zs_mesh = ys_mesh - xs_mesh

    fig, ax = subplots()
    ax.quiver(xs_mesh, ys_mesh, us, vs)
    axis('equal')

    savefig(environment.get_name() + "_field")


def trajectory_learning(environment):
    f = p.figure()
    p.axis([0, 1, 0, 1])

    for i in range(len(environment.starting_points)):
        player_one = Player()
        player_two = Player()
        player_one.set_q_table([environment.starting_points[i][0], environment.starting_points[i][1]])
        player_two.set_q_table([environment.starting_points_two[i][0], environment.starting_points_two[i][1]])

        q_table_two, q_table_two, prob_one, prob_two\
            = independent_learning(environment, player_one, player_two)
        p.plot(prob_one[:], prob_two[:], color='black')

    p.xlabel('Player 1, probability of playing ' + environment.get_first_action_name())
    p.ylabel('Player 2, probability of playing ' + environment.get_first_action_name())
    f.savefig(environment.get_name() + "_trajectory")


def opponent_modelling(environment):
    # prob_one = []
    # prob_two = []
    #
    # for k in range(nr_episodes):
    #     environment.set_action_player_one(player_one.get_action(k))
    #     environment.set_action_player_two(player_two.get_action(k))
    #
    #     player_one.update_q_table(environment.get_action_player_one(), environment.get_reward_player_one())
    #     player_two.update_q_table(environment.get_action_player_two(), environment.get_reward_player_two())
    #
    #     prob_one.append(player_one.get_probability_action(0, k))
    #     prob_two.append(player_two.get_probability_action(0, k))
    #
    # return player_one.get_q_table(), player_two.get_q_table(), prob_one, prob_two

    nr_episodes = 1000

    player_one = DQNPlayer()
    player_two = DQNPlayer()

    for k in range(nr_episodes):
        action_one = player_one.get_action()
        action_two = player_two.get_action()

        environment.set_action_player_one(action_one)
        environment.set_action_player_two(action_two)

        reward_one = environment.get_reward_player_one()
        reward_two = environment.get_reward_player_two()

        player_one.update_replay_memory(action_one, reward_one)
        player_two.update_replay_memory(action_two, reward_two)


def main():

    print("Matching Pennies Environment")
    mpe = MatchingPenniesEnvironment()
    dynamics_learning(mpe)
    trajectory_learning(mpe)

    print("Prisoner's Dilemma Environment")
    pde = PrisonersDilemmaEnvironment()
    dynamics_learning(pde)
    trajectory_learning(pde)


if __name__ == "__main__":
    main()
