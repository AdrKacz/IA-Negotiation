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
        self.state_space = [f'{i + 1}.{j}' for i in range(self.price_space_size) for j in range(self.time_space_size)] \
        + [f'v.{i + 1}.{j}' for i in range(self.price_space_size) for j in range(self.time_space_size)]

        self.reward_mapping = {}

        self.state = None

class Agent:
    def __init__(self):
        self.num_episodes = 1e4
        self.max_steps_per_episode = 100

        self.learning_rate = 0.1
        self.discount_rate = 0.99

        self.max_exploration_rate = 1
        self.min_exploration_rate = 0.1
        self.exploration_rate_decay = 0.001

        self.action_space_size = len(Environment._instances[Environment].action_space)
        self.state_space_size = len(Environment._instances[Environment].state_space)

        self.q_table = []


    def train(self):
        env = Environment._instances[Environment]



if __name__ == '__main__':
    environment = Environment()
    agent = Agent()
