import curses
import time

from environment import Environment
from buyer import Buyer
from seller import Seller

def main(stdscr):
    # Clear screen

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

    env = Environment(stdscr)
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
