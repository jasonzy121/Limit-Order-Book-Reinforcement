import time
import sys
import numpy as np
import sys
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
import os


from config_GOOG import Config
from replay_buffer import ReplayBuffer
from schedule import LinearSchedule
from message_queue import Message_Queue
from order_queue import Order_Queue
from limit_order_book import Limit_Order_book
from message_queue import Message_Queue
from order_queue import Order_Queue
from dqn_model import DQN
from Neural_Net import Neural_DQN


def evaluate_policy(m, oq, mq):
	rewards = []
	test_start, test_end, order_direction, V, H, T, depth= m._config.test_start, m._config.test_end, m._config.direction,\
	m._config.I, m._config.H, m._config.T, m._config.depth
	episodes, real_times = load_episodes(test_start, test_end, order_direction, H, oq, mq)
	for k in range(len(episodes)):
		print ('I am at the %d episode'%(k))
		real_time = real_times[k]
		states, reward, actions, done_mask = m.simulate_an_episode(V, T, 
			H, real_time, order_direction,
			m.get_best_action_fn(), depth)
		print (reward)
		rewards.append(np.sum(reward))
		# Only append the final reward
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

def main():
	config = Config()
	config.mode = 'test'
	config.dropout = 1.0
	model = Neural_DQN(config)
	#model = DQN(config)
	model.initialize()
	oq = Order_Queue(config.order_path)
	mq = Message_Queue(config.message_path)
	rewards= evaluate_policy(model, oq, mq)
	print(np.mean(rewards))

if __name__ == '__main__':
	main()
