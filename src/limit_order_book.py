import numpy as np

class Limit_Order_book(object):
    """
    Abstract Class for Limit Order Book

    Initialize with the following information:
    1) Initial state of the Limit Order Book, including:
    	Bid prices (Descending order) and the corresponding sizes
    	Ask prices (Ascending order) and the corresponding sizes 
    2) Depth of the Limit order book
    3) Dummy prices to reach depth, dummy for ask price, negative for bid
    4) Our own amount of the stocks to trade
    5) The initial limit order price
    6) Our own trade type: -1 for sell, 1 for buy

    When we need to update our own limit order, use update_own_order(price)

    When get a new limit order from the message, use process(type, size, price, direction)
    """

    _DUMMY_VARIABLE = 9999999999

    def __init__(self, bid=np.empty((0,), dtype=int), 
                bid_size=np.empty((0,), dtype=int),
                ask=np.empty((0,), dtype=int),
                ask_size=np.empty((0,), dtype=int),
                own_amount_to_trade=100,
                own_init_price=9999999999,
                own_trade_type=-1):
        """
        Initializer for LOB
        """

        assert(len(bid)==len(bid_size))
        self.bid = bid
        self.bid_size = bid_size

        assert(len(ask)==len(ask_size))
        self.ask = ask
        self.ask_size = ask_size        

        #Initialize own order info
        self.init_own_order(own_amount_to_trade, own_init_price, own_trade_type)


    def init_own_order(self, own_amount_to_trade, own_init_price, own_trade_type):
        """
        Initializer for own order info
        """
        self.own_price = own_init_price
        self.own_amount_to_trade = own_amount_to_trade
        self.own_trade_type = own_trade_type

        self.own_reward = 0.0
        self.own_earlier_orders = 0 #Total number of limit orders before us, including same price but earlier orders

        #Add our own limit order to the LOB
        if self.own_amount_to_trade > 0:
            self.add_order(self.own_amount_to_trade, self.own_price, self.own_trade_type, own=True)


    def update_own_order(self, price, amount = None):
        """
        Helper to update our own order info, only need the new price
        """
        
        if price != self.own_price or ((amount is not None) and (amount !=self.own_amount_to_trade)): #Only need to update if different price
            if self.own_amount_to_trade > 0:
                self.delete_order(self.own_amount_to_trade, self.own_price, self.own_trade_type, own=True)
            self.own_price = price
            if amount is not None:
                self.own_amount_to_trade = amount
            if self.own_amount_to_trade > 0:
                self.add_order(self.own_amount_to_trade, self.own_price, self.own_trade_type, own=True)


    def process(self, type, size, price, direction):
        """
        Process other limit order messages
        Type 1: new limit order
        Type 2 or 3: cancellation or deletion of a limit order, here we assume deleting the earliest ones of the same price
        Type 4: Execution of a visible limit order, equivalent to adding a new limit order of the opposite direction, and
        	then execute the matching
        Type 5: Execution of a hidden limit order, ignored since unobservable
        """
        if type == 1:
            self.add_order(size, price, direction, own=False)
        elif type == 2 or type == 3:
            self.delete_order(size, price, direction, own=False)
        elif type == 4:
            self.add_order(size, price, -direction, own=False)


    def add_order(self, size, price, direction, own=False):
        """
        Execute the matching first, and then insert the remaining ones
        """
        executed = self.partial_execution(size, price, direction, own)
        if executed < size:
            self.insert_order(size - executed, price, direction, own)

    def delete_order(self, size, price, direction, own=False, cancel=True):
        """
        Delete order from the LOB and update number of orders before our own order
        """
        if size <= 0:
            return 0

        if direction == -1: #delete sell order, check ask
            index = np.searchsorted(self.ask, price) #self.ask is in ascending order
            if cancel and not own:
                try:
                    assert(self.ask[index] == price)
                    if self.own_price == price and direction == self.own_trade_type:
                        assert(self.ask_size[index] - self.own_amount_to_trade >= size)
                    else:
                        assert(self.ask_size[index] >= size)
                except:
                    return 0
            else:
                assert(self.ask[index] == price)
                assert(self.ask_size[index] >= size)


            if self.ask_size[index] == size: # have to remove the entry and add dummy if remove whole order
                self.ask = np.delete(self.ask, index)
                self.ask_size = np.delete(self.ask_size, index)
            else:
                self.ask_size[index] -= size

            if price < self.own_price:
                self.own_earlier_orders -= size
            elif price == self.own_price:
                if not own:
                    #if same price as our own order, only remove the earlier ones
                    else_executed = min(size, self.own_earlier_orders - np.sum(self.ask_size[:index]))
                    self.own_earlier_orders -= else_executed
                    if not cancel:
                        own_executed = min(size - else_executed, self.own_amount_to_trade) 
                        self.own_amount_to_trade -= own_executed
                        return own_executed

        elif direction == 1: #delete buy order, check bid
            index = self.bid.size - np.searchsorted(self.bid[::-1], price, side='right') #self.bid is in descending order
            if cancel and not own:
                try:
                    assert(self.bid[index] == price)
                    if self.own_price == price and direction == self.own_trade_type:
                        assert(self.bid_size[index] - self.own_amount_to_trade >= size)
                    else:
                        assert(self.bid_size[index] >= size)
                except:
                    return 0
            else:
                assert(self.bid[index] == price)
                assert(self.bid_size[index] >= size)

            if self.bid_size[index] == size:
                self.bid = np.delete(self.bid, index)
                self.bid_size = np.delete(self.bid_size, index)
            else:
                self.bid_size[index] -= size

            if price > self.own_price:
                self.own_earlier_orders -= size
            elif price == self.own_price:
                if not own:
                    else_executed = min(size, self.own_earlier_orders - np.sum(self.bid_size[:index]))
                    self.own_earlier_orders -= else_executed
                    if not cancel:
                        own_executed = min(size - else_executed, self.own_amount_to_trade) 
                        self.own_amount_to_trade -= own_executed
                        return own_executed
        return 0

    def insert_order(self, size, price, direction, own=False):
        """
        Insert order to the LOB and update number of orders before our own order
        """
        if direction == -1: #insert sell order, check ask
            index = np.searchsorted(self.ask, price)
            extra = 0 #track number of existing same price ones
            if index == len(self.ask) or self.ask[index] != price: #need to insert new entry
                self.ask = np.insert(self.ask, index, price)
                self.ask_size = np.insert(self.ask_size, index, size)
            else:
                extra = self.ask_size[index]
                self.ask_size[index] += size

            if not own: #update number of earlier orders
                if direction == self.own_trade_type and price < self.own_price:
                    self.own_earlier_orders += size
            else: #calculate number of earlier orders
                self.own_earlier_orders = np.sum(self.ask_size[:index]) + extra

        elif direction == 1: #insert buy order, check bid
            index = self.bid.size - np.searchsorted(self.bid[::-1], price, side='right')
            extra = 0
            if index == len(self.bid) or self.bid[index] != price:
                self.bid = np.insert(self.bid, index, price)
                self.bid_size = np.insert(self.bid_size, index, size)
            else:
                extra = self.bid_size[index]
                self.bid_size[index] += size
                
            if not own:
                if direction == self.own_trade_type and price > self.own_price:
                    self.own_earlier_orders += size
            else:
                self.own_earlier_orders = np.sum(self.bid_size[:index]) + extra


    def partial_execution(self, size, price, direction, own=False):
        """
        Match the new order to the LOB and return executed number of orders
        """
        remaining = size #remaining number of orders to execute
        order_reward = 0.0 #reward from executing this order
        own_executed = 0

        if direction == -1: #sell order, check bid
            while remaining > 0 and len(self.bid) > 0 and self.bid[0] >= price:
                to_execute = min(remaining, self.bid_size[0])
                order_reward += to_execute * self.bid[0]
                own_executed += self.delete_order(to_execute, self.bid[0], 1, own=own, cancel=False) #remove matched order
                remaining -= to_execute

        elif direction == 1: #buy order, check ask
            while remaining > 0 and len(self.ask) > 0 and self.ask[0] <= price:
                to_execute = min(remaining, self.ask_size[0])
                order_reward -= to_execute * self.ask[0]
                own_executed += self.delete_order(to_execute, self.ask[0], -1, own=own, cancel=False)
                remaining -= to_execute

        executed = size - remaining

        if own:
            self.own_amount_to_trade -= executed
            self.own_reward += order_reward
        elif direction != self.own_trade_type and executed > self.own_earlier_orders:
        	#Calculate number of our own limit order that got executed
            self.own_reward += self.own_price * own_executed * direction

        return executed

    def get_mid_price(self):
        if self.bid[0] < self._DUMMY_VARIABLE and self.ask[0] > -self._DUMMY_VARIABLE:
            return (self.ask[0] - self.bid[0]) // 2 + self.bid[0]
        if self.bid[0] < self._DUMMY_VARIABLE:
            return self.bid[0]
        return self.ask[0]

    def display_book(self, level):
        bid = np.pad(self.bid, [0, max(0,level-self.bid.size)], 'constant', constant_values=-self._DUMMY_VARIABLE)[:level][:,np.newaxis]
        bid_size = np.pad(self.bid_size, [0, max(0,level-self.bid_size.size)], 'constant', constant_values=0)[:level][:,np.newaxis]
        ask = np.pad(self.ask, [0, max(0,level-self.ask.size)], 'constant', constant_values=self._DUMMY_VARIABLE)[:level][:,np.newaxis]
        ask_size = np.pad(self.ask_size, [0, max(0,level-self.ask_size.size)], 'constant', constant_values=0)[:level][:,np.newaxis]
        return np.concatenate([bid, bid_size, ask, ask_size], axis=1)