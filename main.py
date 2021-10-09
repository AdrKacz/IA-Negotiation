import curses
import time

stdscr = None

class Environment:
    def __init__(self):
        self.buyer = None
        self.seller = None

    def update(self):
        assert self.buyer.min_price <= self.buyer.price <= self.buyer.max_price
        assert self.seller.min_price <= self.seller.price <= self.seller.max_price

    def display(self, y_shift=2):
        if not self.buyer or not self.seller:
            return

        # Shift
        x_shift = len('SELLER ')

        # Shift between min and max
        min_price, max_price = min(self.buyer.min_price, self.seller.min_price), max(self.buyer.max_price, self.seller.max_price)
        def local_transform(price):
            assert 0 <= price <= max_price
            return int((price - min_price) / (max_price - min_price) * (curses.COLS - 1 - x_shift) + x_shift)

        # Metadata
        # ----- Scale
        for i in range(min_price, max_price + 1, 10):
            stdscr.addstr(y_shift, local_transform(i) - len(str(i)) + 1, str(i), curses.color_pair(0))
        # ----- Label
        stdscr.addstr(y_shift + 1, 0, 'BUYER', curses.color_pair(0))
        stdscr.addstr(y_shift + 2, 0, 'SELLER', curses.color_pair(0))
        # ----- Background
        min_buyer, max_buyer = local_transform(self.buyer.min_price), local_transform(self.buyer.max_price)
        stdscr.addstr(y_shift + 1, min_buyer, (max_buyer - min_buyer) * ' ', curses.color_pair(1))

        min_seller, max_seller = local_transform(self.seller.min_price), local_transform(self.seller.max_price)
        stdscr.addstr(y_shift + 2, min_seller, (max_seller - min_seller) * ' ', curses.color_pair(3))

        # Buyer Price
        x = local_transform(self.buyer.price)
        stdscr.addch(y_shift + 1, x, ' ', curses.color_pair(2))

        # Seller price
        x = local_transform(self.seller.price)
        stdscr.addstr(y_shift + 2, x, ' ', curses.color_pair(4))

        stdscr.refresh()

class Buyer:
    def __init__(self):
        self.min_price = 0
        self.max_price = 80
        self.price = 30

    def act(self):
        self.price = min(self.max_price, self.price + 10)

class Seller:
    def __init__(self):
        self.min_price = 20
        self.max_price = 100
        self.price = 50

    def act(self):
        self.price = max(self.min_price, self.price - 10)


def main(local_stdscr):
    # Clear screen
    global stdscr
    stdscr = local_stdscr

    curses.curs_set(0) # invisible cursor
    curses.use_default_colors()
    stdscr.clear()


    scaled = lambda x: int(1000 * x / 255)
    curses.use_default_colors()
    curses.init_pair(0, 0, -1) # Base
    curses.init_color(1, scaled(191), scaled(255), scaled(234)) # Light Green
    curses.init_color(2, scaled(0), scaled(179), scaled(120)) # Green
    curses.init_color(3, scaled(255), scaled(192), scaled(177)) # Light Red
    curses.init_color(4, scaled(179), scaled(45), scaled(12)) # Red

    curses.init_pair(1, -1, 1)
    curses.init_pair(2, -1, 2)
    curses.init_pair(3, -1, 3)
    curses.init_pair(4, -1, 4)

    stdscr.clear()

    env = Environment()
    buyer, seller = Buyer(), Seller()
    env.buyer, env.seller = buyer, seller
    env.display()
    for i in range(10):
        time.sleep(1)
        stdscr.addstr(0, 0, f'Iteration {i}\t')
        stdscr.refresh()

        buyer.act()
        env.update()
        seller.act()
        env.update()
        env.display()

    stdscr.addstr(0, 0, 'Press any key to quit\t')
    stdscr.refresh()
    stdscr.getkey()



if __name__ == '__main__':
    curses.wrapper(main)
