class MatchingPenniesEnvironment:

    def __init__(self):
        self.action_player_one = 0
        self.action_player_two = 0
        self.payoff_matrix_a = [[1, 0], [0, 1]]
        self.payoff_matrix_b = [[0, 1], [1, 0]]

    def get_payoff_matrix(self, player):
        if player == 0:
            return self.payoff_matrix_a
        return self.payoff_matrix_b

    def set_action_player_one(self, action):
        self.action_player_one = action

    def set_action_player_two(self, action):
        self.action_player_two = action

    def get_action_player_one(self):
        return self.action_player_one

    def get_action_player_two(self):
        return self.action_player_two

    def get_reward_player_one(self):
        return self.payoff_matrix_a[self.get_action_player_one()][self.get_action_player_two()]

    def get_reward_player_two(self):
        return self.payoff_matrix_b[self.get_action_player_one()][self.get_action_player_two()]
