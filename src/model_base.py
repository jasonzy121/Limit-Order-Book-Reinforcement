import time
import sys
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
		self._eps_schedule = LinearSchedule(
			self._config.eps_begin,
			self._config.eps_end,
			self._config.nsteps)
		self._lr_schedule = LinearSchedule(
			self._config.lr_begin,
			self._config.lr_end,
			self._config.lr_nsteps)
		self._oq = Order_Queue(self._config.order_path)
		self._mq = Message_Queue(self._config.message_path)
		self._bf = ReplayBuffer(1000000, config)

		self._action_fn = self.get_action_fn()

		self.build()

	def build(self):
		pass

	def initialize(self):
		pass

	def get_random_action(self, state):
		pass

	def get_best_action(self, state):
		### return action, q value
		pass

	def get_action(self, state):
		if np.random.random() < self._eps_schedule.get_epsilon():
			return self.get_random_action(state)[0]
		else:
			return self.get_best_action(state)[0]

	def get_random_action_fn(self):
		def random_action_fn(t, amount, state, mid_price):
			action = np.random.randint(self._config.L) # action = L for market order
			price = (action-self._config.L//2) * self._config.base_point + mid_price
			return (price, action)
		return random_action_fn

	def get_action_fn(self):
		def action_fn(t, amount, state, mid_price):
			action = self.get_action(state)
			price = (action-self._config.L//2) * self._config.base_point + mid_price
			return (price, action)
		return action_fn

	def pad_state(self, states, state_history):
		tmp_states, tmp_its = zip(*states)
		tmp_state = np.concatenate([np.expand_dims(state, -1) for state in tmp_states], axis=-1)
		tmp_state = np.pad(tmp_state, ((0,0),(0,0),(state_history-tmp_state.shape[-1],0)), 'constant', constant_values=0)
		tmp_it = tmp_its[-1]
		return ([tmp_state], [tmp_it])

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
			state = self.process_state(state)
			states.append(state)

			mid_price = lob.get_mid_price()
			state_input = self.pad_state(states[-self._config.state_history:], self._config.state_history)
			price, action = action_fn(start_time+H-t, amount_remain, state_input, mid_price)
			actions.append(action)
			done_mask.append(False)

			lob.update_own_order(price, amount_remain)

			for idx, message in self._mq.pop_to_next_time(t+dH):
				lob.process(**message)
				if lob.own_amount_to_trade == 0:
					done_mask.append(True)
					state = (lob.display_book(depth), np.array([0, 1.0*(start_time+H-self._mq._time)/H], dtype=float))
					state = self.process_state(state)
					states.append(state)
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
			state = self.process_state(state)
			states.append(state)
			done_mask.append(False)

			lob.update_own_order(lob.own_trade_type*Limit_Order_book._DUMMY_VARIABLE)
			if lob.own_amount_to_trade == 0:
				rewards.append(lob.own_reward - cum_reward)
			else:
				rewards.append(-Limit_Order_book._DUMMY_VARIABLE)
			tmp1 = 1.0 * lob.own_amount_to_trade / amount
			state = (lob.display_book(depth), np.array([tmp1, 0], dtype=float))
			state = self.process_state(state)
			states.append(state)
			actions.append(self._config.L)
			done_mask.append(True)
		return (states, rewards, actions, done_mask[1:])

	def sampling_buffer(self):
		for start_time in range(self._config.train_start, self._config.train_end, self._config.H):
			states, rewards, actions, done_mask = self.simulate_an_episode(
				self._config.I, self._config.T, self._config.H, start_time, 
				self._config.direction, self._action_fn, self._config.depth)
			self._bf.store(states, actions, rewards, done_mask)

	def process_state(self, state):
		state_book, state_it = state
		state_book = state_book.astype('float32')
		state_book[:,0] /= 1.e6
		state_book[:,1] /= 1.e2
		state_book[:,2] /= 1.e6
		state_book[:,3] /= 1.e2
		return (state_book, state_it)

if __name__ == '__main__':
	config = Config()
	m = model(config)
	states, rewards, actions, done_mask = m.simulate_an_episode(m._config.I, m._config.T, 
		m._config.H, m._config.train_start, m._config.direction,
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

	# states, it, states_p, it_p, actions, rewards, done_mask = bf.sample(2)
	# print(states)
	# print(it)
	# print(states_p)
	# print(it_p)
	# print(actions)
	# print(rewards)
	# print(done_mask)

