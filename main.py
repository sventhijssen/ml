import ternary
from matplotlib.pyplot import show, subplots, axis, savefig, figure, plot as plt
from numpy import meshgrid, array, arange, zeros, matrix, linspace
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

    print(len(strategies))

    xs_mesh = []
    ys_mesh = []
    for strat in strategies:
        xs_mesh.append(0.5*((strat[0]+2*strat[2])/(strat[0]+strat[1]+strat[2])))
        ys_mesh.append((math.sqrt(3)/2)*((strat[0])/(strat[0]+strat[1]+strat[2])))

    fig = plt.figure
    plt.plot(xs_mesh,ys_mesh)


    #us = array(dynamics.get_mesh_dynamics(xs_mesh, ys_mesh, 0))
    #vs = array(dynamics.get_mesh_dynamics(xs_mesh, ys_mesh, 1))
    #zs_mesh = ys_mesh - xs_mesh

    #fig, ax = subplots()
    #ax.quiver(xs_mesh, ys_mesh, us, vs)
    #axis('equal')

    #savefig(environment.get_name() + "_field")

    # combs = []
    # for xs in strategies:
    #     for ys in strategies:
    #         combs.append((xs, ys))

    combs = [[0.8, 0.1, 0.1]]
    for xs in strategies:
        combs.append((xs, [0.8, 0.1, 0.1]))

    print(len(combs))

    xs = [array([0.1, 0.1, 0.8])]
    ys = [array([0.1, 0.6, 0.3])]

    for i in range(100):
        x = dynamics.get_change(xs[i], ys[i], 0)
        y = dynamics.get_change(xs[i], ys[i], 1)
        x = array(list((map(lambda k: k+1/3, x))))
        y = array(list((map(lambda k: k+1/3, y))))
        xs.append(x)
        ys.append(y)

    for j in xs:
        print(j)



    #trial and error :/
    # plot_results = []
    # for i in range(0, len(results)):
    #     print(results[i])
    #     plot_results.append(array(list((map(lambda k: k+1/3, results[i])))))
    #
    # print("------------")
    # for j in range(len(plot_results)):
    #     print(plot_results[j])

    #TODO: make combinations for the same strategies for player A and player B

    figure, tax = ternary.figure()
    tax.scatter(strategies, marker='s', color='red', label="Red Squares")
    #tax.plot(plot_results, linewidth=2.0, label="Curve")
    #tax.scatter(plot_results, marker='s', color='blue', label="Blue Squares")
    tax.plot(xs, marker='s', color='blue', label="Player X")
    tax.plot(ys, marker='s', color='red', label="Player Y")
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
