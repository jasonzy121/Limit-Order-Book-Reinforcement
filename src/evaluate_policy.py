import numpy as np
import copy

from limit_order_book import Limit_Order_book
from message_queue import Message_Queue
from order_queue import Order_Queue

def evaluate_policy(test_start, test_end, order_direction, V, H, T, oq, mq, action):
	rewards = []
	episodes, real_times = load_episodes(test_start, test_end, order_direction, H, oq, mq)
	for k in range(len(episodes)):
		episode = episodes[k] 
		real_time = real_times[k]
		rewards.append(simulate_reward(episode, V, T, H, action, real_time, mq))
	return rewards


def load_episodes(test_start, test_end, order_direction, H, oq, mq):
	lob_data, time = read_order_book(test_start, test_end, H, oq, mq)
	lob = [Limit_Order_book(**lob_data, own_amount_to_trade = 0, 
					own_init_price=-order_direction*Limit_Order_book._DUMMY_VARIABLE,
					own_trade_type=order_direction) for lob_data in lob_data]
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


def simulate_reward(lob, amount, T, H, action, time, mq):
	"""
	simulate to next state, we need to calculate the remaining inventory given the current i and price a, and the immediate reward
	(revenue from the executed orders)
	"""
	mq.reset()
	mq.jump_to_time(time)

	lob_copy = copy.deepcopy(lob)

	for t in range(time, time + H, H//T):
		price = action(time + H - t, amount, lob_copy)
		lob_copy.update_own_order(price, amount)

		for idx, message in mq.pop_to_next_time(t + H/T):
			lob_copy.process(**message)
			if lob_copy.own_amount_to_trade == 0:
				return lob_copy.own_reward

		amount = lob_copy.own_amount_to_trade

	lob_copy.update_own_order(lob_copy.own_trade_type*Limit_Order_book._DUMMY_VARIABLE)
	if lob_copy.own_amount_to_trade > 0 and lob_copy.own_trade_type == 1:
		return -Limit_Order_book._DUMMY_VARIABLE
	else:
		return lob_copy.own_reward


