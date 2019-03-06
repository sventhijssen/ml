from itertools import product


class MatchingPenniesEnvironment:

    def __init__(self):
        self.PlayerOneAction = 0
        self.PlayerTwoAction = 0



    @staticmethod
    def get_actions():
        return ['H', 'T']

    def get_number_of_actions(self):
        return len(self.get_actions())

    def get_states(self):
        return list(product(self.get_actions(), self.get_actions()))

    def get_number_of_states(self):
        return len(self.get_states())

    def get_reward_even(self):
        return

    def action_player_one(self,action):
        self.PlayerOneAction = action

    def  action_player_two(self,action):
        self.PlayerTwoAction = action

    #player one is the even player
    def reward_player_one(self):
        if(self.PlayerOneAction == self.PlayerTwoAction):
            return 1
        return -1

    #player two is the uneven player
    def reward_player_two(self):
        if (self.PlayerOneAction != self.PlayerTwoAction):
            return 1
        return -1
