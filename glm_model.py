	    
#Need to make big GLM class, and make exp, poisson, and gaussian all inherit from it. 
#get data from allen institute (need also script for easy downloading of data)
import tensorflow as tf

from glm_utils import *

import pyprind

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
		lr = 1e-2, eps = 1e-4, bias_init = 3, train_params = True,
		reg = 'l1', alpha = .1, non_lin = tf.exp, scale_init = 1, verbose = True):
		'''
		Initialize the computational graph for the exponential GLM. 
		
		'''	

		self.non_lin, self.alpha, self.reg, self.verbose = non_lin, alpha, reg, verbose
		self.max_iters, self.lr, self.eps = max_iters, lr, eps

		self.sess = tf.Session()
		
		self.design_ = tf.placeholder('float32', [None, N])
		self.obs_ = tf.placeholder('float32', [None, 1])
		self.weights = tf.Variable(weight_init, dtype = 'float32')
		self.offset = tf.Variable(bias_init, dtype = 'float32', trainable = train_params)
		self.scale = tf.Variable(scale_init, dtype = 'float32', trainable = train_params)
		self.log_loss = self._logloss()

		self.sess.run(tf.initialize_all_variables())

	def _logloss(self):
		'''

		'''	

		alpha = self.alpha

		fx = tf.matmul(self.design_, self.weights) - self.offset

		lam = self.non_lin(fx) 
		lam_ = tf.mul(self.scale,lam)+ self.eps
		

		coef = tf.log(lam_)
		distrib = tf.div(self.obs_, lam_)
		self.loss = tf.reduce_sum(coef + distrib)

		if self.reg == 'l2':	
			self.loss += alpha*tf.reduce_sum(tf.matmul(self.weights, self.weights, transpose_a = True))
			self.loss += alpha*tf.reduce_sum(tf.pow(self.scale, 2))
			self.loss += alpha*tf.reduce_sum(tf.pow(self.offset, 2))

		if self.reg == 'l1': 
			self.loss += alpha*tf.reduce_sum(self.weights + self.offset + self.scale )
		return self.loss


	def fit(self, X, y, X_val, y_val, batch_size = 1000, non_lin = tf.exp):

		loss_arr = []
		cross_val = []

		T = X.shape[0]

		if self.verbose:
			bar = pyprind.ProgBar(self.max_iters, bar_char='*')

		for i in range(self.max_iters):
		    idx = np.random.randint(0, T, size = batch_size)
		    
		    train_feat = X[idx] 
		    train_obs = y[idx]

		    if self.verbose:
		    	bar.update()
		    
		    l, _ = self.sess.run([self.log_loss, self.train_step], feed_dict = {self.design_:train_feat, self.obs_: train_obs[:, np.newaxis]})
		    
		    if i % 10 == 0:

		    	l1 = self.sess.run([self.log_loss], feed_dict = {self.design_:X_val, self.obs_:y_val[:, np.newaxis]})
		    	cross_val.append(l1)
		    	    
		    loss_arr.append(l)
		    

		return loss_arr, cross_val 



	def get_params(self):
		'''
		returns weights, offset, scale
		'''
		return self.weights.eval(self.sess), self.offset.eval(self.sess), self.scale.eval(self.sess)
	
	def predict(self, X):
		'''
		This should run a forward model with the current parameters. 
		returns y (draw from distrib), and lam (conditional intensity)

		Currently implemented for either exponential or sigmoidal non-linearities. 

		Note: I could implement this in tensorflow...

		'''

		weights, offset, scale = self.get_params()

		if self.non_lin == tf.sigmoid:
			h = X.dot(weights)
			lam = scale*sigmoid(h - offset)		
			y = np.random.gamma(1, lam)

		if non_lin == tf.exp:
			h = X.dot(weights)
			lam = scale*np.exp(h - offset)
			y = np.random.gamma(1, lam)

		return y, lam

	def score(self, X, y):
		'''
		MSE between conditional intensity predicted given current params and design matrix X, 
		and observations y. 
		'''
		_, cond = self.predict(X)

		return np.mean((cond - y)**2)



class poisson_GLM():
	def __init__(self, N, weight_init, max_iters = 100,
		lr = 1e-2, eps = 1e-4, bias_init = 3, train_params = True,
		reg = 'l1', alpha = .1, non_lin = tf.exp, scale_init = 1, verbose = True):
		'''
		initializes the computational graph for a poisson GLM with either exponential
		or sigmoidal non-linearity. 
		'''	

		self.non_lin, self.alpha, self.reg, self.verbose = non_lin, alpha, reg, verbose
		self.max_iters, self.lr, self.eps = max_iters, lr, eps
		
		self.sess = tf.Session()
		
		self.design_ = tf.placeholder('float32', [None, N])
		self.obs_ = tf.placeholder('float32', [None, 1])
		self.weights = tf.Variable(weight_init, dtype = 'float32')
		self.offset = tf.Variable(bias_init, dtype = 'float32', trainable = train_params)
		self.scale = tf.Variable(scale_init, dtype = 'float32', trainable = train_params)

		self.log_loss = self._logloss()
		self.train_step =  tf.train.AdamOptimizer(self.lr).minimize(self.log_loss)
		self.sess.run(tf.initialize_all_variables())

	def _logloss():
		'''
		Poisson _logloss
		'''

		alpha = self.alpha

		fx = tf.matmul(self.design_, self.weights) - self.offset

		lam = self.non_lin(fx) 
		lam_ = tf.mul(self.scale,lam)+ self.eps
		

		coef = tf.mul(self.obs_, tf.log(lam_))

		distrib = lam_
		self.loss = tf.reduce_sum(distrib - coef)

		if self.reg == 'l2':	
			self.loss += alpha*tf.reduce_sum(tf.matmul(self.weights, self.weights, transpose_a = True))
			self.loss += alpha*tf.reduce_sum(tf.pow(self.scale, 2))
			self.loss += alpha*tf.reduce_sum(tf.pow(self.offset, 2))

		if self.reg == 'l1': 
			self.loss += alpha*tf.reduce_sum(self.weights + self.offset + self.scale )
		return self.loss


		return None

	def fit(self, X, y, X_val, y_val, batch_size = 1000, non_lin = tf.exp):
		loss_arr = []
		cross_val = []

		T = X.shape[0]

		if self.verbose:
			bar = pyprind.ProgBar(self.max_iters, bar_char='*')

		for i in range(self.max_iters):
		    idx = np.random.randint(0, T, size = batch_size)
		    
		    train_feat = X[idx] 
		    train_obs = y[idx]

		    if self.verbose:
		    	bar.update()
		    
		    l, _ = self.sess.run([self.log_loss, self.train_step], feed_dict = {self.design_:train_feat, self.obs_: train_obs[:, np.newaxis]})
		    
		    if i % 10 == 0:

		    	l1 = self.sess.run([self.log_loss], feed_dict = {self.design_:X_val, self.obs_:y_val[:, np.newaxis]})
		    	cross_val.append(l1)
		    	    
		    loss_arr.append(l)
	    
		return loss_arr, cross_val 


	def get_params(self):
		'''
		returns weights, offset, scale
		'''
		return self.weights.eval(self.sess), self.offset.eval(self.sess), self.scale.eval(self.sess)
	

	def predict(self, X):
		'''
		This should run a forward model with the current parameters. 
		returns y (draw from distrib), and lam (conditional intensity)

		Currently implemented for either exponential or sigmoidal non-linearities. 

		Note: I could implement this in tensorflow...

		'''
	def score(X, y):
		'''
		MSE between predicted conditional intensity given the design matrix X, 
		and emperical observations y. 
		'''
		return None

class gaussian_GLM():
	def __init__(self, N, weight_init, max_iters = 100,
		lr = 1e-2, eps = 1e-4, bias_init = 3, train_params = True,
		reg = 'l1', alpha = .1, non_lin = tf.sigmoid, scale_init = 1, verbose = True):
		'''
		initializes the computational graph for a poisson GLM with either exponential
		or sigmoidal non-linearity. 
		'''	

		self.non_lin, self.alpha, self.reg, self.verbose = non_lin, alpha, reg, verbose
		self.max_iters, self.lr, self.eps = max_iters, lr, eps
		
		self.sess = tf.Session()
		
		self.design_ = tf.placeholder('float32', [None, N])
		self.obs_ = tf.placeholder('float32', [None, 1])
		self.weights = tf.Variable(weight_init, dtype = 'float32')
		self.offset = tf.Variable(bias_init, dtype = 'float32', trainable = train_params)
		self.scale = tf.Variable(scale_init, dtype = 'float32', trainable = train_params)

		self.log_loss = self._logloss()
		self.train_step =  tf.train.AdamOptimizer(self.lr).minimize(self.log_loss)
		self.sess.run(tf.initialize_all_variables())

	def _logloss(self):
		'''
		Gaussian Log loss
		'''

		alpha = self.alpha

		fx = tf.matmul(self.design_, self.weights) - self.offset

		lam = self.non_lin(fx) 
		lam_ = tf.mul(self.scale,lam)+ self.eps
		
		self.loss = tf.reduce_sum(tf.pow(self.obs_ - lam_, 2))

		if self.reg == 'l2':	
			self.loss += alpha*tf.reduce_sum(tf.matmul(self.weights, self.weights, transpose_a = True))
			self.loss += alpha*tf.reduce_sum(tf.pow(self.scale, 2))
			self.loss += alpha*tf.reduce_sum(tf.pow(self.offset, 2))

		if self.reg == 'l1': 
			self.loss += alpha*tf.reduce_sum(self.weights + self.offset + self.scale )
		return self.loss


		return None

	def fit(self, X, y, X_val, y_val, batch_size = 1000, non_lin = tf.exp):
		loss_arr = []
		cross_val = []

		T = X.shape[0]

		if self.verbose:
			bar = pyprind.ProgBar(self.max_iters, bar_char='*')

		for i in range(self.max_iters):
		    #idx = np.random.randint(0, T, size = batch_size)
		    
		    train_feat = X 
		    train_obs = y

		    if self.verbose:
		    	bar.update()
		    
		    l, _ = self.sess.run([self.log_loss, self.train_step], feed_dict = {self.design_:train_feat, self.obs_: train_obs[:, np.newaxis]})
		    
		    if i % 10 == 0:

		    	l1 = self.sess.run([self.log_loss], feed_dict = {self.design_:X_val, self.obs_:y_val[:, np.newaxis]})
		    	cross_val.append(l1)
		    	    
		    loss_arr.append(l)
	    
		return loss_arr, cross_val 


	def get_params(self):
		'''
		returns weights, offset, scale
		'''
		return self.weights.eval(self.sess), self.offset.eval(self.sess), self.scale.eval(self.sess)
	

	def predict(self, X):
		'''
		This should run a forward model with the current parameters. 
		returns y (draw from distrib), and lam (conditional intensity)

		Currently implemented for either exponential or sigmoidal non-linearities. 

		Note: I could implement this in tensorflow...

		'''
	def score(X, y):
		'''
		MSE between predicted conditional intensity given the design matrix X, 
		and emperical observations y. 
		'''
		return None



