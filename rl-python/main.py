from random import random, randrange
from math import exp

class Singleton(type):
    # Use cls._instances[cls] to get instance if any
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class Environment(metaclass=Singleton):
    def __init__(self):
        # Complexity : price_space_size = P , time_space_size = T
        self.price_space_size = 5
        self.time_space_size = 5

        # Actions (P + 2)
        # 1 <= i <= n : offer ; 0 : accept : -1 : reject
        self.action_space = [i + 1 for i in range(self.price_space_size)] + [0, -1]

        self.action_space_size = len(self.action_space)

        # States (P * T + 2 * T + 1)
        # price.time (for each price and for each time)
        # validate.price.time (for each price and for each time)
        # reject.price.time (for each price and for each time)
        # Action 0 -> validate.price.time
        # Action i -> validate.i.time if env (the other agent) accept the offer
        # Start state s
        offer_states = [f'{i + 1}.{j}' for i in range(self.price_space_size) for j in range(self.time_space_size)]
        validate_states = [f'v.{j}' for j in range(self.time_space_size)]
        reject_states = [f'r.{j}' for j in range(self.time_space_size)]
        self.state_space = ['s'] + offer_states + validate_states + reject_states

        self.state_space_size = len(self.state_space)

        self.reward_mapping = {}

        self.state = None
        self.time_step = 0

        # Buyer and Seller
        self.buyer, self.seller = Agent(), Agent()

    def train(self):
        # Initialise Q-Tables
        self.seller.initialise_q_table(self.action_space_size, self.state_space_size)

        self.buyer.initialise_q_table(self.action_space_size, self.state_space_size)

        # Q-Learning Algorithm
        for episode in range(Agent.num_episodes):
            self.reset()
            for step in range(2 * self.time_space_size):
                agent_from, agent_to = self.seller, self.buyer
                if step % 2 == 1:
                    agent_from, agent_to = agent_to, agent_from
                action_index = agent_from.act()
                step_return = self.step(self.action_space[action_index])
                agent_to.update_state(step_return)

                if step_return['done']:
                    break

                if step % 2 == 1:
                    self.time_step += 1
            self.seller.exploration_decay(episode)
            self.buyer.exploration_decay(episode)

    def reset(self):
        self.time_step = 0

        self.seller.reset()
        self.seller.state = self.state_space.index('s')
        self.seller.current_reward = 0

        self.buyer.reset()
        self.buyer.state = self.state_space.index('s')
        self.buyer.current_reward = 0

    def step(self, action):
        # TODO: Dissociate Reward for Seller and Buyer
        # NOTE: Reward based on previous state to maximise profit

        new_state_string, reward, done, info = None, 0, False, ''

        new_state_string = f'{self.time_step}'
        if action == -1:
            new_state_string = f'r.{new_state_string}'
            done = True
        elif action == 0:
            new_state_string = f'v.{new_state_string}'
            done = True
        else:
            new_state_string = f'{action}.{new_state_string}'

        return {
        'new_state': self.state_space.index(new_state_string),
        'reward': reward,
        'done': done,
        'info': info,
        }

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

if __name__ == '__main__':
    environment = Environment()
    environment.train()
