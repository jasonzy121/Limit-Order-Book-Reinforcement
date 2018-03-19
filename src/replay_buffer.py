import numpy as np

class ReplayBuffer(object):
	def __init__(self, size, config):
		self.config = config
		self.size = size
		self.last_idx = -1
		self.history_size = 0

		self.states_stack = np.empty([self.size]+list(self.config.state_shape)+[self.config.state_history+1], dtype=np.float32)
		self.its = np.empty([self.size, 2, 2], dtype=np.float32)
		self.actions = np.empty([self.size], dtype=np.int32)
		self.rewards = np.empty([self.size], dtype=np.float32)
		self.done_mask = np.empty([self.size], dtype=np.bool)

	def process_rewards(self, rewards):
		rewards_processed = []
		for reward in rewards:
			if reward == -9999999999:
				reward = -1000.0
			else:
				reward = reward * 1.e-10
			rewards_processed.append(reward)
		return rewards_processed

	def store(self, states, actions, rewards, done_mask):
		rewards = self.process_rewards(rewards)
		for idx in range(len(actions)):
			self.last_idx += 1
			if self.last_idx == self.size:
				self.last_idx = 0
			self.actions[self.last_idx] = actions[idx]
			self.done_mask[self.last_idx] = done_mask[idx]
			self.rewards[self.last_idx] = rewards[idx]
			tmp = states[max(idx-self.config.state_history+1,0):(idx+2)]
			tmp_states, tmp_its = zip(*tmp)
			tmp_state = np.concatenate([np.expand_dims(state, -1) for state in tmp_states], axis=-1)
			tmp_it = np.concatenate([np.expand_dims(it, -1) for it in tmp_its[-2:]], axis=-1)
			self.states_stack[self.last_idx] = np.pad(tmp_state, ((0,0),(0,0),(self.config.state_history+1-tmp_state.shape[-1],0)), 'constant', constant_values=0)
			self.its[self.last_idx] = tmp_it
			self.history_size += 1

	def sample(self, batch_size):
		idx = np.arange(min(self.size, self.history_size))
		np.random.shuffle(idx)
		idx_choice = idx[:batch_size]

		states = self.states_stack[idx_choice][:,:,:,:-1]
		states_p = self.states_stack[idx_choice][:,:,:,1:]
		it = self.its[idx_choice][:,:,0]
		it_p = self.its[idx_choice][:,:,1]
		actions = self.actions[idx_choice]
		rewards = self.rewards[idx_choice]
		done_mask = self.done_mask[idx_choice]

		return (states, it, states_p, it_p, actions, rewards, done_mask)