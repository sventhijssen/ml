import ternary
from matplotlib.pyplot import show, subplots, axis, savefig, figure, plot as plt
from numpy import meshgrid, array, arange, zeros, matrix, linspace, any
import math
import matplotlib.pyplot as plt

from Dynamics import Dynamics
from MatchingPenniesEnvironment import MatchingPenniesEnvironment
from PrisonersDilemmaEnvironment import PrisonersDilemmaEnvironment
from Player import Player
import pylab as p

from RockPaperScissorsEnvironment import RockPaperScissorsEnvironment
from TernaryDynamics import TernaryDynamics


def independent_learning(environment, player_one, player_two):
    nr_episodes = 100000

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

    #TODO: make combinations for the same strategies for player A and player B

    figure, tax = ternary.figure()
    tax.scatter(strategies, marker='s', color='red', label="Red Squares")
    #tax.plot(plot_results, linewidth=2.0, label="Curve")
    #tax.scatter(plot_results, marker='s', color='blue', label="Blue Squares")
    # tax.plot(xs, marker='s', color='blue', label="Player X")
    # tax.plot(ys, marker='s', color='red', label="Player Y")
    #X, Y = meshgrid(xs_mesh, xs_mesh)
    #ax.streamplot(X, Y, array(results), array(results))

    # that is a lot!
    #tax.scatter(results, marker='s', color='blue')

    # we should not plot as scatter, but as curve
    #still rubbish results
    #tax.plot(results, linewidth=2.0, label="Curve")

    tax.show()



    # # arange(start, stop, step)
    # xs_mesh, ys_mesh = meshgrid(3, 3)
    #
    # us = array(dynamics.get_mesh_dynamics(xs_mesh, ys_mesh, 0))
    # #vs = array(dynamics.get_mesh_dynamics(xs_mesh, ys_mesh, 1))
    # # zs_mesh = ys_mesh - xs_mesh
    #
    # # fig, ax = subplots()
    # # ax.quiver(xs_mesh, ys_mesh, us, vs)
    # # axis('equal')
    # #
    # # savefig(environment.get_name() + "_field")
    #
    # ## Sample trajectory plot
    # figure, tax = ternary.figure(scale=1.0)
    # figure.set_size_inches(5, 5)
    #
    # tax.boundary()
    # tax.gridlines(multiple=0.2, color="black")
    # tax.set_title("Plotting of sample trajectory data", fontsize=10)
    #
    # # Plot the data
    # tax.plot(us, linewidth=2.0, label="Curve")
    # tax.ticks(axis='lbr', multiple=0.2, linewidth=1, tick_formats="%.1f", offset=0.02)
    #
    # tax.get_axes().axis('off')
    # tax.clear_matplotlib_ticks()
    # tax.legend()
    # tax.show()


def trajectory_learning(environment):
    # f = p.figure()
    # p.axis([0, 1, 0, 1])
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

    # p.xlabel('Player 1, probability of playing ' + environment.get_first_action_name())
    # p.ylabel('Player 2, probability of playing ' + environment.get_first_action_name())
    # f.savefig(environment.get_name() + "_trajectory")
    tax.boundary()
    tax.show()
    #figure.savefig(environment.get_name() + "_trajectory")


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

    print("Rock Paper Scissors Environment")
    rpse = RockPaperScissorsEnvironment()
    dynamics_learning_ternary(rpse)
    trajectory_learning(rpse)


if __name__ == "__main__":
    main()
