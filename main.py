import ternary
import numpy as np
from matplotlib.pyplot import show, subplots, axis, savefig, figure, plot as plt
from numpy import meshgrid, array, arange, zeros, matrix, linspace, any
import math
import matplotlib.pyplot as plt
from random import random

from Dynamics import Dynamics
from FictitiousPlayer import FictitiousPlayer
from QFictitiousPlayer import QFictitiousPlayer
from MatchingPenniesEnvironment import MatchingPenniesEnvironment
from PrisonersDilemmaEnvironment import PrisonersDilemmaEnvironment
from Player import Player
import pylab as p

from RockPaperScissorsEnvironment import RockPaperScissorsEnvironment
from ShapleyRockPaperScissorsEnvironment import ShapleyRockPaperScissorsEnvironment
from TernaryDynamics import TernaryDynamics


def independent_learning(environment, player_one, player_two):
    nr_episodes = 50000

    prob_one = []
    prob_two = []

    for k in range(nr_episodes):
        environment.set_action_player_one(player_one.get_action(k))
        environment.set_action_player_two(player_two.get_action(k))

        player_one.update_q_table(environment.get_action_player_one(), environment.get_reward_player_one())
        player_two.update_q_table(environment.get_action_player_two(), environment.get_reward_player_two())

        prob_one.append(
            [player_one.get_probability_action(0, k),
             player_one.get_probability_action(1, k),
             player_one.get_probability_action(2, k)]
        )
        prob_two.append(
            [player_two.get_probability_action(0, k),
             player_two.get_probability_action(1, k),
             player_two.get_probability_action(2, k)]
        )

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


def to2d(s):
    if not any(s):
        return s
    return 0.5*((s[0]+2*s[2])/(s[0]+s[1]+s[2])), (math.sqrt(3)/2)*((s[0])/(s[0]+s[1]+s[2]))

#transform barycentric coordinates to cartesian coordinates
def tocart(s):
    #the three triangle vertices (top, left and right)
    r_1=[0.5,1]
    r_2=[0,0]
    r_3=[1,0]
    return s[0]*0.5+s[1]*0+s[2]*1 , s[0]*1+s[1]*0+s[2]*0

def dynamics_learning_ternary(environment):
    # DYNAMICS
    dynamics = TernaryDynamics(environment)

    # https://stackoverflow.com/questions/1843194/plotting-vector-fields-in-python-matplotlib

    strategies = []
    for x_i in range(0, 11):
        for y_i in range(0, 11):
            x = x_i/10
            y = y_i/10
            z = (10-x_i-y_i)/10
            if x + y <= 1:
                strategies.append((x, y, z))

    results = []
    for s in strategies:
        change = dynamics.get_change(s)
        results.append(change)

    print(len(results))

    xs_mesh = []
    ys_mesh = []
    for s in strategies:
        x, y = tocart(s)
        xs_mesh.append(x)
        ys_mesh.append(y)

    us = []
    vs = []
    for r in results:
        us.append(tocart(r)[0])
        vs.append(tocart(r)[1])

    fig, ax = subplots()
    ax.quiver(xs_mesh, ys_mesh, us, vs)
    axis('equal')
    fig.show()

    savefig(environment.get_name() + "_field")

    figure, tax = ternary.figure()
    tax.scatter(strategies, marker='s', color='red', label="Red Squares")
    tax.right_corner_label("R", fontsize=12)
    tax.top_corner_label("P", fontsize=12)
    tax.left_corner_label("S", fontsize=12)
    tax.show()

def trajectory_learning(environment):
    figure, tax = ternary.figure()
    for i in range(len(environment.starting_points)):
        player_one = Player(3)
        player_two = Player(3)
        player_one.set_q_table([environment.starting_points[i][0], environment.starting_points[i][1], environment.starting_points[i][2]])
        player_two.set_q_table([environment.starting_points_two[i][0], environment.starting_points_two[i][1], environment.starting_points_two[i][2]])

        q_table_two, q_table_two, prob_one, prob_two\
            = independent_learning(environment, player_one, player_two)
        # p.plot(prob_one[:], prob_two[:], color='black')
        tax.plot(prob_one[:], linewidth=2.0, label="Curve")

    tax.boundary()
    tax.right_corner_label("R", fontsize=12)
    tax.top_corner_label("P", fontsize=12)
    tax.left_corner_label("S", fontsize=12)
    tax.show()
    figure.savefig(environment.get_name() + "_trajectory")

def combined_learning(environment, player_one, player_two):
        nr_episodes = 10000

        prob_one = []
        prob_two = []

        for k in range(nr_episodes):
            environment.set_action_player_one(player_one.get_action(k))
            environment.set_action_player_two(player_two.get_action(k))

            player_one.update_q_table(environment.get_action_player_one(),environment.get_action_player_two(), environment.get_reward_player_one())
            player_two.update_q_table(environment.get_action_player_two(),environment.get_action_player_one(), environment.get_reward_player_two())

            prob_one.append(
                [player_one.get_probability_action(0),
                 player_one.get_probability_action(1),
                 player_one.get_probability_action(2)]
            )
            prob_two.append(
                [player_two.get_probability_action(0),
                 player_two.get_probability_action(1),
                 player_two.get_probability_action(2)]
            )

        return player_one.get_q_table(), player_two.get_q_table(), prob_one, prob_two

def trajectory_learning_combined(environment, shapley):
    # f = p.figure()
    # p.axis([0, 1, 0, 1])
    figure, tax = ternary.figure()
    for i in range(len(environment.starting_points_combined)):
        player_one = QFictitiousPlayer(3)
        player_two = QFictitiousPlayer(3)
        player_one.set_q_table(environment.starting_points_combined[i])
        print(player_one.get_q_table())
        player_two.set_q_table(environment.starting_points_combined[i])
        print(player_two.get_q_table())
        if (shapley):
            vis1, vis2 = np.zeros(shape=(3, 3)),np.zeros(shape=(3, 3))
            vis1[0,0] = 1
            player_one.set_visits(vis1)
            vis2[1,1] = 1
            player_two.set_visits(vis2)
            player_one.set_stages(1)
            player_two.set_stages(1)

        q_table_two, q_table_two, prob_one, prob_two\
            = combined_learning(environment, player_one, player_two)
        # p.plot(prob_one[:], prob_two[:], color='black')
        tax.plot(prob_one[:], linewidth=2.0, label="Curve")

    tax.boundary()
    tax.right_corner_label("R", fontsize=12)
    tax.top_corner_label("P", fontsize=12)
    tax.left_corner_label("S", fontsize=12)

    tax.show()
    figure.savefig(environment.get_name() + "_trajectory_combined")

def fictitious_play(environment, player_one, player_two):
    nr_stages = 1000

    # initial strategy is [1,0,0] for both players

    prob_one = []
    prob_two = []

    for k in range(nr_stages):
        action_player_one = player_one.get_action(environment.get_payoff_matrix(0))
        action_player_two = player_two.get_action(environment.get_payoff_matrix(0))

        player_one.set_action(action_player_two)
        player_two.set_action(action_player_one)

        prob_one.append(player_one.get_probabilities())
        prob_two.append(player_two.get_probabilities())

    return prob_one, prob_two

def trajectory_learning_fictitious(environment,shapley):
    figure, tax = ternary.figure()
    for i in range(len(environment.starting_points)):
        player_one = FictitiousPlayer()
        player_two = FictitiousPlayer()
        player_one.set_probabilities(environment.starting_points[i])
        player_two.set_probabilities(environment.starting_points_two[i])
        if(shapley):
            player_one.set_visits(np.array([1,0,0]))
            player_two.set_visits(np.array([0,1,0]))
            player_one.set_stages(1)
            player_two.set_stages(1)

        prob_one, prob_two\
            = fictitious_play(environment, player_one, player_two)
        print(prob_one)
        tax.plot(prob_one[:], linewidth=3, label="Curve")

    tax.boundary()
    tax.right_corner_label("R", fontsize=12)
    tax.top_corner_label("P", fontsize=12)
    tax.left_corner_label("S", fontsize=12)
    tax.show()
    figure.savefig(environment.get_name() + "_trajectory_fictitious")


def main():

    # print("Matching Pennies Environment")
    # mpe = MatchingPenniesEnvironment()
    # dynamics_learning(mpe)
    # trajectory_learning(mpe)
    #
    # print("Prisoner's Dilemma Environment")
    # pde = PrisonersDilemmaEnvironment()
    # dynamics_learning(pde)
    # trajectory_learning(pde)

    rpse = RockPaperScissorsEnvironment()
    print("Dynamics in Rock Paper Scissors Environment")
    dynamics_learning_ternary(rpse)
    print("Q-Learning in Rock Paper Scissors Environment")
    trajectory_learning(rpse)
    print("Fictitious Learning in Rock Paper Scissors Environment")
    trajectory_learning_fictitious(rpse,False)
    print("Fictitious Learning in Shapley's Rock Paper Scissors Environment")
    srpse = ShapleyRockPaperScissorsEnvironment()
    trajectory_learning_fictitious(srpse,True)

    print("Combined Fictitious and Q-Learning in Rock Paper Scissors Environment")
    trajectory_learning_combined(rpse,False)
    print("Combined Fictitious and Q-Learning in Shapley's Rock Paper Scissors Environment")
    trajectory_learning_combined(srpse,True)

if __name__ == "__main__":
    main()
