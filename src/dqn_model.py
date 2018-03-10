import sys
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers

from config import Config
from model_base import model

class DQN(model):
	"""
	Implement Neural Network with Tensorflow
	"""
	def add_placeholders_op(self):
		state_shape = self._config.state_shape
		state_history = self._config.state_history
		# a state shape is (depth, 4)

		# self.s_book: batch of book states, type = float32
		# self.s_it: batch of inventory and time states, type = float32
		# self.a: batch of actions, type = int32
		# self.r: batch of rewards, type = float32
		# self.sp_book: batch of next book states, type = float32
		# self.sp_it: batch of next inventory and time states, type = float32
		# self.done_mask: bath of done, type = bool
		# self.lr: learning rate, type = float32

		self.s_book = tf.placeholder(dtype=tf.float32, shape=[None, state_shape[0], state_shape[1], state_history])
		self.s_it = tf.placeholder(dtype=tf.float32, shape=[None, 2])
		self.a = tf.placeholder(dtype=tf.int32, shape=[None])
		self.r = tf.placeholder(dtype=tf.float32, shape=[None])
		self.sp_book = tf.placeholder(dtype=tf.float32, shape=[None, state_shape[0], state_shape[1], state_history])
		self.sp_it = tf.placeholder(dtype=tf.float32, shape=[None, 2])
		self.done_mask = tf.placeholder(dtype=tf.bool, shape=[None])
		self.lr = tf.placeholder(dtype=tf.float32, shape=[])

	def get_q_values_op(self, state, scope, reuse=False):
		### Implement a Fully-Connected Network, replace with CNN later
		num_actions = self._config.L + 1 # 1 for market order
		state_book, state_it = state
		with tf.variable_scope(scope, reuse=reuse):
			state_book_flattened = layers.flatten(state_book)
			state_out = tf.concat([state_book_flattened, state_it], axis=1)

			out = layers.fully_connected(state_out, num_actions, activation_fn=None)
		return out

	def add_update_target_op(self, q_scope, target_q_scope):
		q_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=q_scope)
		target_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=target_q_scope)
		self.update_target_op = tf.group(*[tf.assign(var1, var2) for var1, var2 in zip(target_var, q_var)])

	def add_loss_op(self, q, target_q):
		num_actions = self._config.L + 1 # 1 for market order
		Q_samp = self.r + self._config.gamma * tf.reduce_max(target_q, axis=1) * (1 - tf.cast(self.done_mask, tf.float32))
		Q_s_a = tf.reduce_sum(tf.one_hot(self.a, num_actions) * q, axis=1)
		self.loss = tf.reduce_mean(tf.square(Q_samp - Q_s_a))

	def add_optimizer_op(self, scope):
		optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
		var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
		grads = optimizer.compute_gradients(self.loss, var_list=var_list)
		if self._config.grad_clip:
			grads = [(tf.clip_by_norm(grad, self._config.clip_val), var) for grad, var in grads]
		self.train_op = optimizer.apply_gradients(grads)
		self.grad_norm = tf.global_norm([grad for grad, _ in grads])

	def build(self):
		self.add_placeholders_op()

		q_state = (self.s_book, self.s_it)
		self.q = self.get_q_values_op(q_state, scope='q', reuse=False)

		target_q_state = (self.sp_book, self.sp_it)
		self.target_q = self.get_q_values_op(target_q_state, scope='target_q', reuse=False)

		self.add_update_target_op('q','target_q')

		self.add_loss_op(self.q, self.target_q)

		self.add_optimizer_op('q')

	def initialize(self):
		self.sess = tf.Session()
		self.sess.run(tf.global_variables_initializer())
		self.sess.run(self.update_target_op)
		self.saver = tf.train.Saver()

	def train(self):
		self.sampling_buffer()

		t = 0

		total_loss = 0
		while t < self._config.nsteps_train:
			t += 1
			self._lr_schedule.update(t)
			self._eps_schedule.update(t)
			loss_t = self.train_step(t, self._config.batch_size, self._lr_schedule.get_epsilon())
			total_loss += loss_t
			if t % self._config.print_freq == 0:
				sys.stdout.write('Iter {} \t Loss {} \n'.format(t, total_loss / t))
				sys.stdout.flush()

	def train_step(self, t, batch_size, lr):
		states, it, states_p, it_p, actions, rewards, done_mask = self._bf.sample(batch_size)
		feed_dict = {self.s_book:states, self.s_it:it, self.sp_book:states_p, self.sp_it:it_p, 
			self.a:actions, self.r:rewards, self.done_mask:done_mask, self.lr:lr}
		loss_eval, _ = self.sess.run([self.loss, self.train_op], feed_dict=feed_dict)

		if t % self._config.target_update_freq == 0:
			self.sess.run(self.update_target_op)
		if t % self._config.saving_freq == 0:
			if not os.path.exists(self._config.model_output):
				os.makedirs(self._config.model_output)
			self.saver.save(self.sess, self._config.model_output)
		if t % self._config.simulation_freq == 0:
			self.sampling_buffer()

		return loss_eval

	def get_random_action(self, state):
		action = np.random.randint(self._config.L)
		q = self.get_q_values(state)[0]
		q_value = q_value = q[action]
		return (action, q_value)

	def get_best_action(self, state):
		q = self.get_q_values(state)[0]
		action = np.argmax(q[:-1])
		q_value = q[action]
		return (action, q_value)

	def get_q_values(self, state):
		state_book, state_it = state
		q, = self.sess.run([self.q], feed_dict={self.s_book:state_book, self.s_it:state_it})
		return q

if __name__ == '__main__':
	config = Config()
	model = DQN(config)
	model.initialize()
	model.train()
