from matplotlib.pyplot import show, subplots, axis, savefig, figure
from numpy import meshgrid, array, arange, zeros, matrix

from Dynamics import Dynamics
from MatchingPenniesEnvironment import MatchingPenniesEnvironment
from PrisonersDilemmaEnvironment import PrisonersDilemmaEnvironment
from Player import Player
import pylab as p


def independent_learning(environment):
    nr_episodes = 1000000

    player_one = Player()
    player_two = Player()

    for k in range(nr_episodes):
        environment.set_action_player_one(player_one.get_action(k))
        environment.set_action_player_two(player_two.get_action(k))

        player_one.update_q_table(environment.get_action_player_one(), environment.get_reward_player_one())
        player_two.update_q_table(environment.get_action_player_two(), environment.get_reward_player_two())

    return player_one.get_q_table(), player_two.get_q_table()


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
    nr_episodes = 100000

    pairs = [(0,0 ), (0,1), (1,0), (1,1), (0.1, 0.9), (0.34, 0.72), (0.8, 0.12)] # (0.20, 0.80), (0.40, 0.80), (0.60, 0.80), (0.80,0.20), (0.80, 0.40), (0.80, 0.20)]
    f = p.figure()
    p.axis([0,1,0,1])

    pairs_one = [(0.1,0.9),(0.2,0.8),(0.3,0.7),(0.4,0.6),(0.5,0.5),(0.6,0.4),(0.7,0.3),(0.8,0.2),(0.9,0.1),(0,0),(0,1),(1,0)]
    pairs_two = pairs_one

    for i in range(len(pairs_one)):
        player_one = Player()
        player_two = Player()
        player_one.set_q_table([pairs_one[i][0], pairs_one[i][1]])
        player_two.set_q_table([pairs_two[i][0], pairs_two[i][1]])
        prob_one = []
        prob_two = []

        for k in range(nr_episodes):
            environment.set_action_player_one(player_one.get_action(k))
            environment.set_action_player_two(player_two.get_action(k))

            player_one.update_q_table(environment.get_action_player_one(), environment.get_reward_player_one())
            player_two.update_q_table(environment.get_action_player_two(), environment.get_reward_player_two())

            prob_one.append(player_one.get_probability_action(0, k))
            prob_two.append(player_two.get_probability_action(0, k))

        p.plot(prob_one[:], prob_two[:], color='black')
    p.xlabel('Player 1, probability of playing ' + environment.get_first_action_name())
    p.ylabel('Player 2, probability of playing ' + environment.get_first_action_name())
    f.savefig(environment.get_name() + "_trajectory")


def main():
    n = 100
    print("Matching Pennies Environment")
    #player_one = zeros(2)
    #player_two = zeros(2)
    #choice_one = zeros(2)
    #choice_two = zeros(2)
    #for i in range(n):
    #    mpe = MatchingPenniesEnvironment()
    #    a, b = independent_learning(mpe)
    #    if a[0] > a[1]:
    #        choice_one[0] += 1
    #    else:
    #        choice_one[1] += 1
    #    if b[0] > b[1]:
    #        choice_two[0] += 1
    #    else:
    #        choice_two[1] += 1
    #    player_one += a
    #    player_two += b
    #player_one.dot(1/n)
    #player_two.dot(1/n)
    #print(player_one)
    #print(player_two)
    #print("Number of final choices (highest q-value is picked)")
    #print(choice_one)
    #print(choice_two)
    mpe = MatchingPenniesEnvironment()
    dynamics_learning(mpe)
    trajectory_learning(mpe)

    print("Prisoner's Dilemma Environment")
    #player_one = zeros(2)
    #player_two = zeros(2)
    #choice_one = zeros(2)
    #choice_two = zeros(2)
    #for i in range(n):
    #    pde = PrisonersDilemmaEnvironment()
    #    a, b = independent_learning(pde)
    #    print(a)
    #    print(b)
    #    if a[0] > a[1]:
    #        choice_one[0] += 1
    #        print("DING")
    #    else:
    #        choice_one[1] += 1
    #    if b[0] > b[1]:
    #        choice_two[0] += 1
    #        print("DING")
    #    else:
    #        choice_two[1] += 1
    #    player_one += a
    #    player_two += b
    #player_one.dot(1/n)
    #player_two.dot(1/n)
    #print(player_one)
    #print(player_two)
    #print("Number of final choices (highest q-value is picked)")
    #print(choice_one)
    #print(choice_two)
    pde = PrisonersDilemmaEnvironment()
    dynamics_learning(pde)
    trajectory_learning(pde)


if __name__ == "__main__":
    main()
