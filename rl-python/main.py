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

        # States (2 * P * T)
        # price.time (for each price and for each time)
        # validate.price.time (for each price and for each time)
        # Action 0 -> validate.price.time
        # Action i -> validate.i.time if env (the other agent) accept the offer
        self.state_space = [f'{i + 1}.{j}' for i in range(self.price_space_size) for j in range(self.time_space_size)] + [f'v.{i + 1}.{j}' for i in range(self.price_space_size) for j in range(self.time_space_size)]

        self.reward_mapping = {}

        self.state = None

        # Buyer and Seller
        buyer, seller = Agent('buyer'), Agent('seller')

    def train(self):
        # TODO: Implement training of both Buyer and Seller
        # offer = seller.get_offer()
        # if offer <= 0: break
        # buyer.set_offer(offer)
        # offer = buyer.get_offer()
        # if offer <= 0: break
        # seller.set_offer(offer)
        pass

    def reset(self):
        # TODO: Implement reset
        self.state = 0
        return self.state

    def step(self, action):
        # TODO: Different seller / buyer
        return {
        'new_state': 0,
        'reward': 0,
        'done': False,
        'info': '',
        }

class Agent:
    def __init__(self, type):
        assert type in ['seller', 'buyer']

        self.num_episodes = int(1e4)
        self.max_steps_per_episode = 100

        self.learning_rate = 0.1
        self.discount_rate = 0.99

        self.max_exploration_rate = 1
        self.min_exploration_rate = 0.1
        self.exploration_rate_decay = 0.001

        self.action_space_size = len(Environment._instances[Environment].action_space)
        self.state_space_size = len(Environment._instances[Environment].state_space)

        self.q_table = [[0 for j in range(self.action_space_size)] for i in range(self.state_space_size)]


    def train(self):
        # TODO: Concurrent learning seller / agent
        env = Environment._instances[Environment]

        # Re-Initialise Q-Table
        for i in range(len(self.q_table)):
            for j in range(len(self.q_table[i])):
                self.q_table[i][j] = 0

        # Q-Learning Algorithm
        rewards_all_episodes = list()
        exploration_rate = self.max_exploration_rate

        for episode in range(self.num_episodes):
            state = env.reset()
            reward_current_episode = 0
            for step in range(self.max_steps_per_episode):
                action_index = None
                # Exploration versus Exploitation
                if random() > exploration_rate: # Exploitation
                    action_index = self.exploit()
                else:
                    action_index = randrange(0, self.action_space_size)
                action = env.action_space[action_index]

                step_return = env.step(action)

                new_state = step_return['new_state']

                # Update Q-table Q(state, action)
                self.q_table[state][action_index] = self.q_table[state][action_index] * (1 - self.learning_rate) + self.learning_rate * (step_return['reward'] + self.discount_rate * max(self.q_table[new_state]))

                state = new_state
                if step_return['done']:
                    reward_current_episode += step_return['reward']
                    break

            # exploration_rate decay
            exploration_rate = self.min_exploration_rate + (self.max_exploration_rate - self.min_exploration_rate) * exp(-self.exploration_rate_decay * episode)
            rewards_all_episodes.append(reward_current_episode)

        return rewards_all_episodes

    def exploit(self):
        state = Environment._instances[Environment].state
        action_index = self.q_table[state].index(max(self.q_table[state]))
        return action_index


if __name__ == '__main__':
    environment = Environment()
    environment.train()
