import time
import numpy as np

from config import Config
from replay_buffer import ReplayBuffer
from schedule import LinearSchedule
from message_queue import Message_Queue
from order_queue import Order_Queue
from limit_order_book import Limit_Order_book

class model(object):
	def __init__(self, config):
		self._config = config
		self._schedule = LinearSchedule(
			self._config.eps_begin,
			self._config.eps_end,
			self._config.nsteps)
		self._oq = Order_Queue(self._config.order_path)
		self._mq = Message_Queue(self._config.message_path)

		self.build()

	def build(self):
		pass

	def initialize(self):
		pass

	def get_random_action(self):
		pass

	def get_best_action(self, state):
		### return action, q value
		pass

	def get_action(self, state):
		if np.random.random() < self._schedule.get_epsilon():
			return self.get_random_action()[0]
		else:
			return self.get_best_action(state)[0]

	def get_random_action_fn(self):
		def random_action_fn(t, amount, state, mid_price):
			action = np.random.randint(self._config.L) + 1 # action = 0 for market order
			price = (action-self._config.L//2) * self._config.base_point + mid_price
			return (price, action)
		return random_action_fn

	def simulate_an_episode(self, amount, T, H, start_time, order_direction, action_fn, depth):
		dH = H // T
		self._mq.reset()
		lob_data = self._oq.create_orderbook_time(start_time, self._mq)
		lob = Limit_Order_book(**lob_data, own_amount_to_trade=0,
			own_init_price=-order_direction*Limit_Order_book._DUMMY_VARIABLE,
			own_trade_type=order_direction)
		rewards = []
		states = []
		actions = []
		done_mask = []
		
		amount_remain = amount
		cum_reward = 0

		for t in range(start_time, start_time+H-dH, dH):
			tmp1 = 1.0 * amount_remain / amount # amount remain
			tmp2 = 1.0 * (start_time + H - t) / H # time remain
			state = (lob.display_book(depth), np.array([tmp1, tmp2], dtype=float))
			states.append(state)

			mid_price = lob.get_mid_price()
			price, action = action_fn(start_time+H-t, amount_remain, state, mid_price)
			actions.append(action)
			done_mask.append(False)

			lob.update_own_order(price, amount_remain)

			for idx, message in self._mq.pop_to_next_time(t+dH):
				lob.process(**message)
				if lob.own_amount_to_trade == 0:
					done_mask.append(True)
					states.append((lob.display_book(depth), np.array([0, 1.0*(start_time+H-self._mq._time)/H], dtype=float)))
					rewards.append(lob.own_reward - cum_reward)
					break
			if done_mask[-1]:
				break
			else:
				rewards.append(lob.own_reward - cum_reward)
				cum_reward = lob.own_reward
				amount_remain = lob.own_amount_to_trade

		if not done_mask[-1]:
			tmp1 = 1.0 * amount_remain / amount
			tmp2 = 1.0 * (start_time + H - t - dH) / H
			state = (lob.display_book(depth), np.array([tmp1, tmp2], dtype=float))
			states.append(state)
			done_mask.append(False)

			lob.update_own_order(lob.own_trade_type*Limit_Order_book._DUMMY_VARIABLE)
			if lob.own_amount_to_trade == 0:
				rewards.append(lob.own_reward - cum_reward)
			else:
				rewards.append(lob.own_trade_type*Limit_Order_book._DUMMY_VARIABLE)
			tmp1 = 1.0 * lob.own_amount_to_trade / amount
			state = (lob.display_book(depth), np.array([tmp1, 0], dtype=float))
			states.append(state)
			actions.append(0)
			done_mask.append(True)
		return (states, rewards, actions, done_mask[1:])

if __name__ == '__main__':
	config = Config()
	m = model(config)
	states, rewards, actions, done_mask = m.simulate_an_episode(m._config.I, m._config.T, 
		m._config.H, m._config.start_time, m._config.direction,
		m.get_random_action_fn(), m._config.depth)

	print(states)
	print(rewards)
	print(actions)
	print(done_mask)

	bf = ReplayBuffer(3, config)
	bf.store(states, actions, rewards, done_mask)
	print(bf.states_stack)
	print(bf.its)
	print(bf.actions)
	print(bf.rewards)
	print(bf.done_mask)

	states, it, states_p, it_p, actions, rewards, done_mask = bf.sample(2)
	print(states)
	print(it)
	print(states_p)
	print(it_p)
	print(actions)
	print(rewards)
	print(done_mask)

