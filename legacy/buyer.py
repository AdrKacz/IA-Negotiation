class Buyer:
    def __init__(self, parent):
        self.parent = parent
        self.min_price = 0
        self.max_price = 80
        self.price = None
        self.current_seller = None

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
