from random import random, randrange
from math import exp

class Agent:
    num_episodes = int(1e4)
    max_steps_per_episode = 100

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
        if not self.state:
            return
        new_state = step_return['new_state']

        # Update Q-table Q(state, action)
        self.q_table[self.state][self.last_action_index] = self.q_table[self.state][self.last_action_index] * (1 - self.learning_rate) + self.learning_rate * (step_return['reward'] + self.discount_rate * max(self.q_table[new_state]))

        self.state = new_state

    def act(self):
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

    def exploit(self):
        state = Environment._instances[Environment].state
        action_index = self.q_table[state].index(max(self.q_table[state]))
        return action_index

    def print_q_table(self):
        if not self.q_table:
            return
        for row in self.q_table:
            print(' '.join(map(lambda x:f'{x:.2f}', row)))


class Buyer(Agent):
    pass

class Seller(Agent):
    pass
