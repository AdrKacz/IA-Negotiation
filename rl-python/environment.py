from agent import Buyer, Seller
from copy import deepcopy
from math import exp
import matplotlib.pyplot as plt

class Environment():
    def __init__(self):
        # Training Information
        self.num_cycles = int(5e1)
        self.num_episodes = int(1e2)
        self.num_episodes_testing = int(1e3)


        # Complexity : price_space_size = P , time_space_size = T
        self.price_space_size = 5
        self.time_space_size = 5

        # Actions (P + 2)
        # 1 <= i <= n : offer ; 0 : accept : -1 : reject
        self.action_space = [i + 1 for i in range(self.price_space_size)] + [0, -1]

        self.action_space_size = len(self.action_space)

        # States (P * T + 2)
        # price.time (for each price and for each time)
        # Start state s
        offer_states = [f'{price + 1}.{time}' for time in range(self.time_space_size) for price in range(self.price_space_size)]
        self.state_space = ['s'] + offer_states + ['d']

        self.state_space_size = len(self.state_space)

        self.last_offer = None
        self.time_step = 0

        # Buyer and Seller
        self.buyer, self.seller = Buyer(), Seller()

    def train(self, verbose=False, display_normalized=False, display_plot=False, train_both=True, train_agent=None):
        # Train Seller then Buyer alternatively
        if not train_both and not train_agent:
            raise ValueError('Specified a agent to train if you don\'t train both')
        elif not train_both and train_agent not in ['Seller', 'Buyer']:
            raise ValueError('Train Agent must be either Seller or Buyer')

        # Initialise Q-Tables
        self.seller.initialise_q_table(self.action_space_size, self.state_space_size)
        self.buyer.initialise_q_table(self.action_space_size, self.state_space_size)

        # Statistics
        transactions_validated, transactions_rejected = 0, 0
        seller_wallets, buyer_wallets = list(), list()
        # Q-Learning Algorithm
        trainee, trainer = self.seller, self.buyer
        trainee_wallets, trainer_wallets = seller_wallets, buyer_wallets
        trainee_q_table, trainer_q_table = deepcopy(trainee.q_table), deepcopy(trainer.q_table)

        if train_agent == 'Buyer':
            trainee, trainer = trainer, trainee
            trainee_wallets, trainer_wallets = trainer_wallets, trainee_wallets
            trainee_q_table, trainer_q_table = trainer_q_table, trainee_q_table

        def trainer_first():
            # Trainer Start Negociation
            action_index = trainer.exploit(overwrite_q_table=trainer_q_table)
            step_return = self.step(self.action_space[action_index], trainee)
            trainee.state = step_return['new_state']

            if step_return['done']:
                raise ValueError('Must do an offer at first')

        for cycle in range(self.num_cycles):
            trainee.reset_wallet(), trainer.reset_wallet()
            for episode in range(self.num_episodes):
                self.reset()
                is_trainer_first = episode % 2 == 0
                if is_trainer_first:
                    trainer_first()
                for step in range(self.time_space_size):
                    action_index = trainee.act()
                    step_return = self.step(self.action_space[action_index], trainee)

                    if step_return['done']:
                        if step_return['info']['type'] == 'validated':
                            transactions_validated += 1
                            trainee.update_wallet(step_return['info']['offer']), trainer.update_wallet(step_return['info']['offer'])
                        elif step_return['info']['type'] == 'rejected':
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
                        if step_return['info']['type'] == 'validated':
                            transactions_validated += 1
                            trainee.update_wallet(step_return['info']['offer']), trainer.update_wallet(step_return['info']['offer'])
                        elif step_return['info']['type'] == 'rejected':
                            transactions_rejected += 1
                        break

                    if not is_trainer_first:
                        self.time_step += 1

                if self.time_step >= self.time_space_size:
                    raise ValueError('No one close the transaction before end')

                trainee.exploration_decay(episode)
            # Update Statistics
            trainee_wallets.append(trainee.wallet), trainer_wallets.append(trainer.wallet)
            trainee.save_statistics_delta(), trainer.save_statistics_delta()
            # Switch Trainee and Trainer (cache trainer copy from previous trainee training)
            if train_both:
                trainer_q_table = deepcopy(trainer.q_table)
                trainee, trainer = trainer, trainee
                trainee_wallets, trainer_wallets = trainer_wallets, trainee_wallets
                trainee_q_table, trainer_q_table = trainer_q_table, trainee_q_table

        assert transactions_validated + transactions_rejected ==  self.num_cycles * self.num_episodes

        if not verbose:
            return

        # Display result
        self.display_statistics(self.num_cycles * self.num_episodes, transactions_validated, transactions_rejected)

        # Raw Q-Tables
        self.display_q_tables()

        # Prob. Q-Tables
        if display_normalized:
            self.display_prob_q_tables()

        # Plot Result
        if display_plot:
            self.display_plot(seller_wallets, buyer_wallets)

    def test(self):
        # Statistics
        self.seller.reset_statistics(), self.buyer.reset_statistics()
        transactions_validated, transactions_rejected = 0, 0

        # Test Seller versus Buyer
        first, second = self.seller, self.buyer
        for episode in range(self.num_episodes_testing):
            self.reset()
            for step in range(self.time_space_size):
                action_index = first.exploit()
                step_return = self.step(self.action_space[action_index])

                if step_return['done']:
                    if step_return['info']['type'] == 'validated':
                        transactions_validated += 1
                        first.update_wallet(step_return['info']['offer']), second.update_wallet(step_return['info']['offer'])
                    elif step_return['info']['type'] == 'rejected':
                        transactions_rejected += 1
                    break

                second.state = step_return['new_state']
                action_index = second.exploit()
                step_return = self.step(self.action_space[action_index])

                if step_return['done']:
                    if step_return['info']['type'] == 'validated':
                        transactions_validated += 1
                        first.update_wallet(step_return['info']['offer']), second.update_wallet(step_return['info']['offer'])
                    elif step_return['info']['type'] == 'rejected':
                        transactions_rejected += 1
                    break

                self.time_step += 1

            if self.time_step >= self.time_space_size:
                raise ValueError('No one close the transaction before end')

            # Switch first and second
            first, second = second, first

        # Update Statistics
        assert transactions_validated + transactions_rejected ==  self.num_episodes_testing

        # Display result
        self.display_statistics(self.num_episodes_testing, transactions_validated, transactions_rejected)

        print(f'Seller Wallet: {self.seller.wallet:4>}')
        print(f'Buyer Wallet: {self.buyer.wallet:4>}')

    def display_statistics(self, transactions_total, transactions_validated, transactions_rejected):
        print('\n', '===== ' * 5)
        print(f'Transactions Total\t: {transactions_total:6d}')
        print(f'Transactions Validated\t: {transactions_validated:6d}')
        print(f'Transactions Rejected\t: {transactions_rejected:6d}')
        print('\t\tSeller\t Buyer\t Total')
        print(f'Validated:\t{self.seller.transactions_validated:6d}\t{self.buyer.transactions_validated:6d}\t{self.seller.transactions_validated + self.buyer.transactions_validated:6d}')
        print(f'Rejected:\t{self.seller.transactions_rejected:6d}\t{self.buyer.transactions_rejected:6d}\t{self.seller.transactions_rejected + self.buyer.transactions_rejected:6d}')

    def display_q_tables(self):
        print('\n', '===== ' * 5)
        print(' ' * 10, 'Seller Raw Q-Table\t\t\t\t \t Buyer Raw Q-Table ')
        # Print action space
        to_string = lambda row : ' '.join(map(lambda x:f'{x:>5}', row))
        print(' ' * 10, to_string(self.action_space), '\t|\t', to_string(self.action_space))
        for i in range(len(self.seller.q_table)):
            state = self.state_space[i]
            seller_row = self.seller.q_table[i]
            buyer_row = self.buyer.q_table[i]
            to_string = lambda row : ' '.join(map(lambda x:f'{x:.0e}' if x != 0 else ' ' * 5, row))

            print(f'[{i:2d}] {state:>3}', '>', to_string(seller_row), '\t|\t', to_string(buyer_row))

    def display_prob_q_tables(self):
        print('\n', '===== ' * 5)
        print(' ' * 10, 'Seller Prob. Q-Table\t\t \t Buyer Prob. Q-Table ')
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

    def display_plot(self, seller_wallets, buyer_wallets):
        cycles = list(range(self.num_cycles))
        plt.figure()
        plt.subplot(311)
        plt.title('Seller')
        plt.plot(cycles, self.seller.transactions_validated_list, 'o-', label='Validated', color='g')
        plt.plot(cycles, self.seller.transactions_rejected_list, 'o-', label='Rejected', color='r')
        plt.legend()

        plt.subplot(312)
        plt.title('Buyer')
        plt.plot(cycles, self.buyer.transactions_validated_list, 'o-', label='Validated', color='g')
        plt.plot(cycles, self.buyer.transactions_rejected_list, 'o-', label='Rejected', color='r')
        plt.legend()

        plt.subplot(313)
        plt.title('Wallet')
        plt.plot(cycles, seller_wallets, 'o-', label='Seller', color='b')
        plt.plot(cycles, buyer_wallets, 'o-', label='Buyer', color='m')
        plt.legend()

        plt.show()

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

        new_state_string, done, info = None, False, {}

        new_state_string = f'{self.time_step}'
        if action == -1:
            new_state_string = 'd'
            done = True
            info = {'type': 'rejected'}
        elif action == 0:
            new_state_string = 'd'
            done = True
            info = {'type': 'validated', 'offer': self.last_offer}
        else:
            self.last_offer = action
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
