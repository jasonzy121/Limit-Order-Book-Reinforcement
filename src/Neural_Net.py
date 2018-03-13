import sys
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
import os

from config import Config
from model_base import model
from dqn_model import DQN

class Neural_DQN(DQN):
	def get_q_values_op(self, state, scope, reuse= False):
		hidden_size= self._config.hidden_size
		num_actions = self._config.L + 1 # 1 for market order
		state_book, state_it = state
		with tf.variable_scope(scope, reuse=reuse):
			state_book_flattened = layers.flatten(state_book)
			state_out = tf.concat([state_book_flattened, state_it], axis=1)
			out_1 = layers.fully_connected(state_out, hidden_size)
			out= layers.fully_connected(out_1, num_actions)
		return out

if __name__ == '__main__':
	config = Config()
	config.model_output= '../output/neural_net'
	model = Neural_DQN(config)
	model.initialize()
	model.train()
	