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
