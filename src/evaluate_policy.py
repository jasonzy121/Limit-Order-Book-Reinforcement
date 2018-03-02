import numpy as np

from limit_order_book import Limit_Order_book
from message_queue import Message_Queue
from order_queue import Order_Queue

def evaluate_policy(test_start, test_end, H, T, oq, mq, action):
	rewards = []
	episodes, real_times = load_episodes(test_start, test_end, H, oq, mq)
	for k in range(len(episodes)):
        episode = episodes[k] 
        real_time = real_times[k]
        rewards = simulate(episode[0], int((i+1)*V/I) , a_price, real_time[0], real_time[0], mq)



def load_episodes(test_start, test_end, H, oq, mq):
    lob_data, time = read_order_book(test_start, test_end, H, oq, mq)
    lob = [Limit_Order_book(**lob_data, own_amount_to_trade = 0, 
                    own_init_price=-args.order_direction*Limit_Order_book._DUMMY_VARIABLE,
                    own_trade_type=args.order_direction) for lob_data in lob_data]
    return lob, time

def read_order_book(test_start, test_end, H, oq, mq):
    """
    read the initial limit order book states from the file
    """
    output = []
    time_output = []
    real_time = test_start
    while real_time < test_end:
        mq.reset()
        output.append(oq.create_orderbook_time(real_time, mq))
        time_output.append(real_time)
        real_time = real_time + H
    return output, time_output
