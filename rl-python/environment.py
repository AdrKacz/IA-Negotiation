from agent import Agent, Buyer, Seller

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
        self.buyer, self.seller = Buyer(), Seller()

    def train(self):
        # Initialise Q-Tables
        self.seller.initialise_q_table(self.action_space_size, self.state_space_size)

        self.buyer.initialise_q_table(self.action_space_size, self.state_space_size)

        transactions_rejected = 0
        transactions_validated = 0
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
                    if step_return['info'] == 'validated':
                        transactions_validated += 1
                    elif step_return['info'] == 'requested':
                        transactions_rejected += 1
                    break

                if step % 2 == 1:
                    self.time_step += 1
            self.seller.exploration_decay(episode)
            self.buyer.exploration_decay(episode)

        # Display result
        print('===== ' * 5)
        print(f'Transactions\t\t: {Agent.num_episodes:5d}')
        print(f'Transactions Validated\t: {transactions_validated:5d}')
        print(f'Transactions Rejected\t: {transactions_rejected:5d}')
        print('===== ' * 5)
        print('      Seller Q-Table\t\t\t\t \t Buyer Q-Table ')
        to_string = lambda row : ' '.join(map(lambda x:f'{x:>4}', row))
        print('   ', ' ', to_string(self.action_space), '\t|\t', to_string(self.action_space))
        for i in range(len(self.seller.q_table)):
            state = self.state_space[i]
            seller_row = self.seller.q_table[i]
            buyer_row = self.buyer.q_table[i]
            to_string = lambda row : ' '.join(map(lambda x:f'{x:.2f}', row))
            print(f'{state:>3}', '>', to_string(seller_row), '\t|\t', to_string(buyer_row))

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
            info = 'rejected'
        elif action == 0:
            new_state_string = f'v.{new_state_string}'
            done = True
            reward = 1
            info = 'validated'
        else:
            reward = 10
            new_state_string = f'{action}.{new_state_string}'

        return {
        'new_state': self.state_space.index(new_state_string),
        'reward': reward,
        'done': done,
        'info': info,
        }
