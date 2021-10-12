import auxilary_functions as aux


class Bid:

    def __init__(self, initial_id, delivery_time, transaction_date, cancel_date, price, volume, side, is_activ=True):

        self.id = initial_id

        self.product_id = aux.getPruductId(delivery_time)
        self.transaction_date = transaction_date
        self.calcel_date = cancel_date

        self.price = price
        self.volume = volume
        self.side = side

        self.is_activ = is_activ
