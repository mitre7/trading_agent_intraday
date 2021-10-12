class Transaction:

    def __init__(self, price, quantity, trans_time, side):
        self.price = price
        self.quantity = quantity
        self.trans_time = trans_time
        self.side = side
