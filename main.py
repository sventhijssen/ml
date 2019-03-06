from MatchingPenniesEnvironment import MatchingPenniesEnvironment
from QLearning import QLearning
from PrisonersDilemmaEnvironment import PrisonersDilemmaEnvironment

def main():
    #environment = MatchingPenniesEnvironment()
    environment = PrisonersDilemmaEnvironment()
    nr_episodes = 10000

    player_a = QLearning(environment)
    player_b = QLearning(environment)
    for ep in range(nr_episodes):
        environment.action_player_one(player_a.action())
        environment.action_player_two(player_b.action())

        player_a.reward(environment.PlayerOneAction, environment.reward_player_one())
        player_b.reward(environment.PlayerTwoAction, environment.reward_player_two())

        print(player_a.q_table)
        print(player_b.q_table)



if __name__ == "__main__":
    main()
