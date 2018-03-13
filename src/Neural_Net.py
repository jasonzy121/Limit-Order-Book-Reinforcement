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
		num_actions = self._config.L + 1 # 1 for market order
		state_book, state_it = state

		with tf.variable_scope(scope, reuse=reuse):
			conv_1 = layers.conv2d(inputs=state_book, num_outputs=4, kernel_size=[3,3], stride=[1,1], activation_fn=tf.nn.relu, padding='same')
			conv_2 = layers.conv2d(inputs=conv_1, num_outputs=4, kernel_size=[3,3], stride=[1,1], activation_fn=tf.nn.relu, padding='same')
			conv_2_flattened = layers.flatten(inputs=conv_2)
			state_out = tf.concat([conv_2_flattened, state_it], axis=1)
			state_out= tf.nn.dropout(state_out, self._config.dropout)
			out = layers.fully_connected(state_out, num_actions, activation_fn=None)
		return out

if __name__ == '__main__':
	config = Config()
	config.model_output= '../output/neural_net'
	model = Neural_DQN(config)
	model.initialize()
	model.train()
	