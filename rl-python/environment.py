from agent import Buyer, Seller
from copy import deepcopy
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
        # Training Information
        self.num_cycles = int(1e1)
        self.num_episodes = int(1e3)


        # Complexity : price_space_size = P , time_space_size = T
        self.price_space_size = 5
        self.time_space_size = 5

        # Actions (P + 2)
        # 1 <= i <= n : offer ; 0 : accept : -1 : reject
        self.action_space = [i + 1 for i in range(self.price_space_size)] + [0, -1]

        self.action_space_size = len(self.action_space)

        # States (P * T + 1)
        # price.time (for each price and for each time)
        # Start state s
        offer_states = [f'{price + 1}.{time}' for time in range(self.time_space_size) for price in range(self.price_space_size)]
        self.state_space = ['s'] + offer_states + ['d']

        self.state_space_size = len(self.state_space)

        self.state = None
        self.time_step = 0

        # Buyer and Seller
        self.buyer, self.seller = Buyer(), Seller()

    def train(self, display_normalized=False):
        # Train Seller then Buyer alternatively

        # Initialise Q-Tables
        self.seller.initialise_q_table(self.action_space_size, self.state_space_size)
        self.buyer.initialise_q_table(self.action_space_size, self.state_space_size)

        transactions_rejected = 0
        transactions_validated = 0
        # Q-Learning Algorithm
        trainee, trainer = self.seller, self.buyer
        trainee_q_table, trainer_q_table = deepcopy(trainee.q_table), deepcopy(trainer.q_table)

        def trainer_first():
            # Trainer Start Negociation
            action_index = trainer.exploit(overwrite_q_table=trainer_q_table)
            step_return = self.step(self.action_space[action_index], trainee)
            trainee.state = step_return['new_state']

            if step_return['done']:
                raise ValueError('Must do an offer at first')

        for cycle in range(self.num_cycles):
            for episode in range(self.num_episodes):
                self.reset()
                is_trainer_first = episode % 2 == 0
                if is_trainer_first:
                    trainer_first()
                for step in range(self.time_space_size):
                    action_index = trainee.act()
                    step_return = self.step(self.action_space[action_index], trainee)

                    if step_return['done']:
                        if step_return['info'] == 'validated':
                            transactions_validated += 1
                        elif step_return['info'] == 'rejected':
                            transactions_rejected += 1
                        trainee.update_state(step_return)
                        break

                    if is_trainer_first:
                        self.time_step += 1

                    trainer.state = step_return['new_state']
                    action_index = trainer.exploit(overwrite_q_table=trainer_q_table)
                    step_return = self.step(self.action_space[action_index], trainee)
                    trainee.update_state(step_return)

                    if step_return['done']:
                        if step_return['info'] == 'validated':
                            transactions_validated += 1
                        elif step_return['info'] == 'rejected':
                            transactions_rejected += 1
                        break

                    if not is_trainer_first:
                        self.time_step += 1

                if self.time_step >= self.time_space_size:
                    raise ValueError('No one close the transaction before end')

                trainee.exploration_decay(episode)
            # Switch Trainee and Trainer (cache trainer copy from previous trainee training)
            trainer_q_table = deepcopy(trainer.q_table)
            trainee, trainer = trainer, trainee
            trainee_q_table, trainer_q_table = trainer_q_table, trainee_q_table

        assert transactions_validated + transactions_rejected ==  self.num_cycles * self.num_episodes
        # Display result
        print('\n', '===== ' * 5)
        print(f'Transactions Total\t: {self.num_cycles * self.num_episodes:6d}')
        print(f'Transactions Validated\t: {transactions_validated:6d}')
        print(f'Transactions Rejected\t: {transactions_rejected:6d}')
        print('\t\tSeller\t Buyer\t Total')
        print(f'Validated:\t{self.seller.transactions_validated:6d}\t{self.buyer.transactions_validated:6d}\t{self.seller.transactions_validated + self.buyer.transactions_validated:6d}')
        print(f'Rejected:\t{self.seller.transactions_rejected:6d}\t{self.buyer.transactions_rejected:6d}\t{self.seller.transactions_rejected + self.buyer.transactions_rejected:6d}')
        print('\n', '===== ' * 5)
        print(' ' * 10, 'Seller Raw Q-Table\t\t\t\t \t Buyer Raw Q-Table ')
        # Print action space
        to_string = lambda row : ' '.join(map(lambda x:f'{x:>5}', row))
        print(' ' * 10, to_string(self.action_space), '\t|\t', to_string(self.action_space))

        # Raw Q-Tables
        for i in range(len(self.seller.q_table)):
            state = self.state_space[i]
            seller_row = self.seller.q_table[i]
            buyer_row = self.buyer.q_table[i]
            to_string = lambda row : ' '.join(map(lambda x:f'{x:.0e}' if x != 0 else ' ' * 5, row))

            print(f'[{i:2d}] {state:>3}', '>', to_string(seller_row), '\t|\t', to_string(buyer_row))

        # Softmax Q-Tables
        if not display_normalized:
            return
        print('\n', '===== ' * 5)
        print(' ' * 10, 'Seller Softmax Q-Table\t\t \t Buyer Softmax Q-Table ')
        # Print action space
        to_string = lambda row : ' '.join(map(lambda x:f'{x:>3}', row))
        print(' ' * 10, to_string(self.action_space), '\t|\t', to_string(self.action_space))

        normalized = lambda row : [v / sum(row) if sum(row) > 0 else 0 for v in row]
        to_string_hidden = lambda row : [f'{v:.2f}' if v > 0 else '   ' for v in normalized(row)]
        to_string = lambda row : ' '.join(map(lambda x:f'{x[1:]:>3}', to_string_hidden(row)))
        for i in range(len(self.seller.q_table)):
            state = self.state_space[i]
            seller_row = self.seller.q_table[i]
            buyer_row = self.buyer.q_table[i]
            print(f'[{i:2d}] {state:>3}', '>', to_string(seller_row), '\t|\t', to_string(buyer_row))

    def reset(self):
        self.time_step = 0

        self.seller.reset()
        self.seller.state = self.state_space.index('s')
        self.seller.current_reward = 0

        self.buyer.reset()
        self.buyer.state = self.state_space.index('s')
        self.buyer.current_reward = 0

    def step(self, action, agent=None):
        # TODO: Dissociate Reward for Seller and Buyer
        # NOTE: Reward based on previous state to maximise profit

        new_state_string, done, info = None, False, ''

        new_state_string = f'{self.time_step}'
        if action == -1:
            new_state_string = 'd'
            done = True
            info = 'rejected'
        elif action == 0:
            new_state_string = 'd'
            done = True
            info = 'validated'
        else:
            new_state_string = f'{action}.{new_state_string}'

        step_return_without_reward = {
        'new_state': self.state_space.index(new_state_string),
        'done': done,
        'info': info,
        }
        return {
            **step_return_without_reward,
            'reward': agent.get_reward(step_return_without_reward) if agent else 0,
        }
