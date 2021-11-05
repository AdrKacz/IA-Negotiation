from random import random, randrange, choice
from math import exp

class Agent:
    learning_rate = 0.1
    discount_rate = 0.99

    max_exploration_rate = 1
    min_exploration_rate = 0.1
    exploration_rate_decay = 0.001

    def __init__(self):
        self.action_space_size = None
        self.state_space_size = None
        self.q_table = None
        self.state = None
        self.last_action_index = None
        self.exploration_rate = Agent.max_exploration_rate

        # Keep statistic of training phase
        self.wallet = 0
        self.transactions_validated = 0
        self.transactions_rejected = 0

        self.transactions_validated_list = list()
        self.transactions_rejected_list = list()


    def initialise_q_table(self, action_space_size, state_space_size):
        self.exploration_rate = Agent.max_exploration_rate
        self.transactions_validated = 0
        self.transactions_rejected = 0
        self.transactions_validated_list = list()
        self.transactions_rejected_list = list()

        self.action_space_size = action_space_size
        self.state_space_size = state_space_size

        self.q_table = [[0 for j in range(action_space_size)] for i in range(state_space_size)]

        self.reset()
        self.reset_wallet()

    def reset(self):
        self.state = None
        self.last_action_index = None

    def update_state(self, step_return):
        if self.last_action_index ==  None:
            raise ValueError('Cannot update Q-Table if no action has been made')


        new_state = step_return['new_state']
        # Update Q-table Q(state, action)
        self.q_table[self.state][self.last_action_index] = self.q_table[self.state][self.last_action_index] * (1 - self.learning_rate) + self.learning_rate * (step_return['reward'] + self.discount_rate * max(self.q_table[new_state]))

        self.state = new_state

    def act(self):
        action_index = None
        # Exploration versus Exploitation
        if random() > self.exploration_rate: # Exploitation
            action_index = self.exploit() # Statistic in Exploit
        else: # Exploration
            from_action, to_action = self.dynamic_action_space()
            action_index = randrange(from_action, to_action)
            self.last_action_index = action_index
            self.update_statistics()
        return action_index

    def exploration_decay(self, episode):
        self.exploration_rate = Agent.min_exploration_rate + (Agent.max_exploration_rate - Agent.min_exploration_rate) * exp(-Agent.exploration_rate_decay * episode)

    def dynamic_action_space(self):
        # Must do an offer at first
        # Must accept or reject if last round
        from_action, to_action = 0, self.action_space_size
        if self.state == 0: # Start
            to_action -= 2 # Remove Validate and Reject
        elif self.state_space_size + 1 - self.action_space_size <= self.state < self.state_space_size - 1: # Last round
            from_action += self.action_space_size - 2 # Remove offer action

        return from_action, to_action

    def update_statistics(self):
        if self.last_action_index == 5:
            self.transactions_validated += 1
        elif self.last_action_index == 6:
            self.transactions_rejected += 1

    def save_statistics_delta(self):
        if len(self.transactions_validated_list) == 0:
            self.transactions_validated_list.append(self.transactions_validated)
        else:
            self.transactions_validated_list.append(self.transactions_validated - sum(self.transactions_validated_list))

        if len(self.transactions_rejected_list) == 0:
            self.transactions_rejected_list.append(self.transactions_rejected)
        else:
            self.transactions_rejected_list.append(self.transactions_rejected - sum(self.transactions_rejected_list))


    def exploit(self, overwrite_q_table=None):
        q_table = self.q_table
        if overwrite_q_table:
            q_table = overwrite_q_table
        # Get one of the max (to avoid bias if not enough trained yet)
        action_indices, action_value = list(), -float('inf')

        from_action, to_action = self.dynamic_action_space()
        for index, value in enumerate(q_table[self.state][from_action:to_action]):
            if value > action_value:
                action_value = value
                action_indices = [index + from_action]
            elif value == action_value:
                action_indices.append(index + from_action)
        # Return one at random
        self.last_action_index = choice(action_indices)
        self.update_statistics()
        return self.last_action_index

    def print_q_table(self):
        if not self.q_table:
            return
        for row in self.q_table:
            print(' '.join(map(lambda x:f'{x:.2f}', row)))


class Buyer(Agent):
    def reset_wallet(self):
        self.wallet = 0

    def update_wallet(self, offer):
        self.wallet -= offer

    def get_reward(self, step_return_without_reward):
        info = step_return_without_reward['info']
        if info.get('type')  == 'validated':
            assert 1 <= info.get('offer', 0) <= 5
            return 1 - info['offer'] / 5
        return 0

    def __str__(self):
        return 'Buyer '

class Seller(Agent):
    def reset_wallet(self):
        self.wallet = 0

    def update_wallet(self, offer):
        self.wallet += offer

    def get_reward(self, step_return_without_reward):
        info = step_return_without_reward['info']
        if info.get('type') == 'validated':
            assert 1 <= info.get('offer', 0) <= 5
            return info['offer'] / 5
        return 0

    def __str__(self):
        return 'Seller'
