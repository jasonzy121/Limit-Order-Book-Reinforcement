class LinearSchedule(object):
	def __init__(self, eps_begin, eps_end, nsteps):
		self._epsilon = eps_begin
		self._eps_begin = eps_begin
		self._eps_end = eps_end
		self._nsteps = nsteps

	def update(self, t):
		alpha = 1.0 * t / self._nsteps
		self._epsilon = max(alpha*self._eps_end+(1-alpha)*self._eps_begin, self._eps_end)

	def get_epsilon(self):
		return self._epsilon