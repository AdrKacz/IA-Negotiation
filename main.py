import curses
import time

stdscr = None

class Environment:
    def __init__(self):
        self.buyer = None
        self.seller = None
        self.deal = False

    def update(self):
        if self.buyer.price != None:
            assert self.buyer.min_price <= self.buyer.price <= self.buyer.max_price
        if self.seller.price != None:
            assert self.seller.min_price <= self.seller.price <= self.seller.max_price

        if self.seller.price != None and self.seller.price == self.buyer.price:
            self.deal = True

    def display(self, y_shift=1):
        if not self.buyer or not self.seller:
            return

        # Shift
        x_shift = len('SELLER ')
        y_buyer, y_seller = 2, 3

        # Shift between min and max
        min_price, max_price = min(self.buyer.min_price, self.seller.min_price), max(self.buyer.max_price, self.seller.max_price)
        def local_transform(price):
            assert 0 <= price <= max_price
            return int((price - min_price) / (max_price - min_price) * (curses.COLS - 1 - x_shift) + x_shift)

        # Metadata
        # ----- Scale
        for i in range(min_price, max_price + 1, 10):
            stdscr.addstr(y_shift + 1, local_transform(i) - len(str(i)) + 1, str(i), curses.color_pair(0))
        # ----- Label
        stdscr.addstr(y_shift + y_buyer, 0, 'BUYER', curses.color_pair(0))
        stdscr.addstr(y_shift + y_seller, 0, 'SELLER', curses.color_pair(0))
        # ----- Background
        min_buyer, max_buyer = local_transform(self.buyer.min_price), local_transform(self.buyer.max_price)
        stdscr.addstr(y_shift + y_buyer, min_buyer, (max_buyer - min_buyer) * ' ', curses.color_pair(1))

        min_seller, max_seller = local_transform(self.seller.min_price), local_transform(self.seller.max_price)
        stdscr.addstr(y_shift + y_seller, min_seller, (max_seller - min_seller) * ' ', curses.color_pair(3))

        # Buyer Price
        if self.buyer.price != None:
            x = local_transform(self.buyer.price)
            stdscr.addch(y_shift + y_buyer, x, ' ', curses.color_pair(2))

        # Seller price
        if self.seller.price != None:
            x = local_transform(self.seller.price)
            stdscr.addstr(y_shift + y_seller, x, ' ', curses.color_pair(4))

        # Deal
        if self.deal:
            stdscr.addstr(y_shift, 0, f'DEAL at {self.seller.price}', curses.color_pair(5))

        stdscr.refresh()

class Buyer:
    def __init__(self, parent):
        self.parent = parent
        self.min_price = 0
        self.max_price = 80
        self.price = None

    def act(self):
        # Look for desired price of the seller
        seller_price = self.parent.seller.price
        if self.price is None: # Initialise
            if seller_price != None:
                self.price = max(self.min_price, (self.min_price + seller_price) // 2)
            else:
                self.price = self.min_price
            return

        # Price is defined
        assert seller_price != None
        self.price = max(self.min_price, (self.price + seller_price) // 2)

class Seller:
    def __init__(self, parent):
        self.parent = parent
        self.min_price = 20
        self.max_price = 100
        self.price = None

    def act(self):
        # Look for desired price of the buyer
        buyer_price = self.parent.buyer.price
        if self.price is None: # Initialise
            if buyer_price != None:
                self.price = min(self.max_price, (self.max_price + buyer_price) // 2)
            else:
                self.price = self.max_price
            return

        # Price is defined
        assert buyer_price != None
        self.price = min(self.max_price, (self.price + buyer_price) // 2)



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

    curses.init_color(5, scaled(48), scaled(255), scaled(25)) # Flashy Light
    curses.init_color(6, scaled(255), scaled(71), scaled(25)) # Flashy Red

    curses.init_pair(1, -1, 1)
    curses.init_pair(2, -1, 2)
    curses.init_pair(3, -1, 3)
    curses.init_pair(4, -1, 4)

    curses.init_pair(5, 5, 6)
    curses.init_pair(6, 6, -1)

    stdscr.clear()

    env = Environment()
    buyer, seller = Buyer(env), Seller(env)
    env.buyer, env.seller = buyer, seller
    env.display()
    for i in range(10):
        stdscr.getkey()
        # time.sleep(0.5)
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
