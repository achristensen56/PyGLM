import numpy as np
import tensorflow as tf
import pyprind
import functools



	    
#from glm_utils import gamma
#need to implement gamma distribution

class exponential_GLM():
	'''
	Parameters:
		max_iters = 100
		eps = 1e-4
		init = {'ols', 'random'}
		lr = 1e-2
		l_algo = {'Adam', ...}
	Attributes:
		y_
		w
		t_loss_arr
		v_loss_arr
	'''

	def __init__(self, N, weight_init, max_iters = 100,
		lr = 1e-2, l_algo = 'Adam', eps = 1e-4, non_lin = tf.exp):
		'''
		
		'''	

		self.max_iters, self.lr, self.l_algo, self.eps = max_iters, lr, l_algo, eps
		self.sess = tf.Session()
		self.design_ = tf.placeholder('float32', [None, N])
		self.obs_ = tf.placeholder('float32', [None, 1])
		self.weights = tf.Variable(weight_init, dtype = 'float32')
		
		self.c = tf.Variable(0.0, dtype = 'float32')

		self.log_loss = self._logloss()

		if self.l_algo == 'Adam':
			self.train_step =  tf.train.AdamOptimizer(self.lr).minimize(self.log_loss)
		else:
			raise("Not implemented yet")

		self.sess.run(tf.initialize_all_variables())


	def _logloss(self):
		'''
		right now exponential non-linearity is hardcoded
		change that soon
		'''
		#non_lin = tf.exp
		#mu = non_lin(tf.matmul(self.design_, self.weights) + 5)
		#lam = tf.div(1.0, mu)
		#loss = -tf.reduce_sum(tf.log(lam) + tf.log(tf.exp( -tf.mul(self.obs_, lam))))

		self.fx = tf.matmul(self.design_, self.weights)
		self.exp = tf.mul(self.obs_, tf.exp(-self.fx + 5))
		self.loss = tf.reduce_sum(self.fx + self.exp)

		lam2 = 100
		self.loss += lam2*tf.reduce_sum(tf.matmul(self.weights, self.weights, transpose_a = True))

		return self.loss


	def fit(self, X, y, batch_size = 1000):

		loss_arr = []
		T = X.shape[0]

		bar = pyprind.ProgBar(self.max_iters, bar_char='*')

		for i in range(self.max_iters):
		    idx = np.random.randint(0, T, size = batch_size)
		    
		    train_feat = X[idx] 
		    train_obs = y[idx]
		    bar.update()
		    
		    l, _ = self.sess.run([self.log_loss, self.train_step], feed_dict = {self.design_:train_feat, self.obs_: train_obs[:, np.newaxis]})
		    
		    if i > 10:
		        if (loss_arr[-1] - l)**2 < self.eps:
		            break
		    
		    
		    loss_arr.append(l)

		    

		return loss_arr


	
	def fit_predict(self):
		'''
		convenience function for fit and then predict.
		'''
		pass

	def score():
		pass
	
	def predict():
		'''
		This should run a forward model with the current parameters. 


		'''
		pass
