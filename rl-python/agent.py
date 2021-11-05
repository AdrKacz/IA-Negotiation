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
        self.current_reward = 0
        self.exploration_rate = Agent.max_exploration_rate


    def initialise_q_table(self, action_space_size, state_space_size):
        self.action_space_size = action_space_size
        self.state_space_size = state_space_size

        self.q_table = [[0 for j in range(action_space_size)] for i in range(state_space_size)]

        self.reset()

    def reset(self):
        self.state = None
        self.last_action_index = None
        self.current_reward = 0
        self.exploration_rate = Agent.max_exploration_rate

    def update_state(self, step_return):
        if self.last_action_index ==  None:
            self.state = step_return['new_state']
            return

        # ### Helper
        # offer_states = [f'{price + 1}.{time}' for time in range(5) for price in range(5)]
        # validate_states = [f'v.{j}' for j in range(5)]
        # reject_states = [f'r.{j}' for j in range(5)]
        # state_space = ['s'] + offer_states + validate_states + reject_states
        # if self.state == 0:
        #     print(f'Update from {state_space[self.state]}', self)
        # else:
        #     print(f'Update from {state_space[self.state]}', self)
        # ###
        new_state = step_return['new_state']
        # Update Q-table Q(state, action)
        self.q_table[self.state][self.last_action_index] = self.q_table[self.state][self.last_action_index] * (1 - self.learning_rate) + self.learning_rate * (step_return['reward'] + self.discount_rate * max(self.q_table[new_state]))

        self.state = new_state

    def act(self):
        # print('Act', self)
        action_index = None
        # Exploration versus Exploitation
        if random() > self.exploration_rate: # Exploitation
            action_index = self.exploit()
        else:
            action_index = randrange(0, self.action_space_size)
        self.last_action_index = action_index
        return action_index

    def exploration_decay(self, episode):
        self.exploration_rate = Agent.min_exploration_rate + (Agent.max_exploration_rate - Agent.min_exploration_rate) * exp(-Agent.exploration_rate_decay * episode)

    def exploit(self, overwrite_q_table=None):
        q_table = self.q_table
        if overwrite_q_table:
            q_table = overwrite_q_table
        # Get one of the max (to avoid bias if not enough trained yet)
        action_indices, action_value = list(), -float('inf')
        for index, value in enumerate(q_table[self.state]):
            if value > action_value:
                action_value = value
                action_indices = [index]
            elif value == action_value:
                action_indices.append(index)
        # Return one at random
        return choice(action_indices)

    def print_q_table(self):
        if not self.q_table:
            return
        for row in self.q_table:
            print(' '.join(map(lambda x:f'{x:.2f}', row)))


class Buyer(Agent):
    def get_reward(self, state_string):
        if state_string == 'd' and self.last_action_index == 5:
            return 1
        return 0
    def __str__(self):
        return 'Buyer'

class Seller(Agent):
    def get_reward(self, state_string):
        if state_string == 'd' and self.last_action_index == 5:
            return 1
        return 0

    def __str__(self):
        return 'Seller'
