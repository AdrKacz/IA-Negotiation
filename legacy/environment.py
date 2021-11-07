import curses

stdscr = None

class Environment:
    def __init__(self, s):
        global stdscr
        stdscr = s
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
