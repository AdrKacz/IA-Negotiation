from environment import Environment

if __name__ == '__main__':
    environment = Environment()
    print('\n\033[1mBoth Trained - Seller First\033[0m')
    environment.train(train_both=False, train_agent='Seller')
    environment.test()
    print('\n\033[1mBoth Trained - Buyer First\033[0m')
    environment.train(train_both=False, train_agent='Buyer')
    environment.test()
