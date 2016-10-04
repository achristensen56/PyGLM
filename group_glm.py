import tensorflow as tf
from glm_utils import *
import pyprind


class GLM():
	'''
	Parameters:
		max_iters = 100
		eps = 1e-4
		init = {'ols', 'random'}
		lr = 1e-2
		l_algo = {'Adam', ...}
		non_lin = tf.* {should be applied elementwise}
		verbose = True
	Attributes:
		y_
		w
		t_loss_arr
		v_loss_arr

	Each class that inherits from GLM should override the _logloss function. 

	'''

	def __init__(self, weight_init,
		lr = 1e-2, eps = 1e-4, bias_init = 0.01, train_params = True,
		reg = 'l1', alpha = .1, non_lin = tf.sigmoid, scale_init = 3, verbose = True, cap_grads = True):
		'''
		Initialize the computational graph for the exponential GLM. 

		weight_init: self.num_features, self.num_neurons initialization for weights
		
		'''	

		self.num_features, self.num_neurons = weight_init.shape

		self.non_lin, self.alpha, self.reg, self.verbose = non_lin, alpha, reg, verbose
		self.lr, self.eps = lr, eps

		tf.reset_default_graph()
		self.sess = tf.Session()
		
		self.design_ = tf.placeholder('float32', [None, self.num_features])
		self.obs_ = tf.placeholder('float32', [None, self.num_neurons])
		
		self.weights = tf.Variable(weight_init, dtype = 'float32', name = 'weights')

		self.offset = bias_init*tf.Variable(np.ones([self.num_neurons]), dtype = 'float32', trainable = train_params)
		self.scale = scale_init*tf.Variable(np.ones([self.num_neurons]), dtype = 'float32', trainable = train_params)
		self.nonlin_offset = bias_init*tf.Variable(np.ones([self.num_neurons]), dtype = 'float32')


		self.log_loss = self._logloss()

		if cap_grads == True:	
			optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
			
			#optimizer = tf.train.GradientDescentOptimizer(learning_rate = self.lr)
			gvs = optimizer.compute_gradients(self.log_loss)
			capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]

			self.train_step = optimizer.apply_gradients(capped_gvs)
		else:
			self.train_step =  tf.train.AdamOptimizer(self.lr).minimize(self.log_loss)
		

		self.sess.run(tf.initialize_all_variables())

	def _logloss():
		pass


	def fit(self, X, y, X_val = None, y_val = None, batch_size = 1000, max_iters = 500):

		loss_arr = []
		cross_val = []

		T = X.shape[0]

		if self.verbose:
			bar = pyprind.ProgBar(max_iters, bar_char='*')

		converged = False
			
		for i in range(max_iters):
		    idx = np.random.randint(0, T, size = batch_size)
		    
		    train_feat = X[idx] 
		    train_obs = y[idx]

		    if self.verbose:
		    	bar.update()
		    
		    l, _ = self.sess.run([self.log_loss, self.train_step], feed_dict = {self.design_:train_feat, self.obs_: train_obs})


		    if i % 10 == 0 and X_val != None:

		    	l1 = self.sess.run(self.log_loss, feed_dict = {self.design_:X_val, self.obs_:y_val})
		    	cross_val.append(l1)
	    	    
		    loss_arr.append(l)

		    if i > 10:
				if (np.average(cross_val[-5:]) - np.average(cross_val[-10:-5])) >  0.001:
					break

		return np.array(loss_arr), np.array(cross_val) 


	def predict_trace(self, X):
		return self.sess.run(self.predict, feed_dict = {self.design_:X})

	def get_params(self):
		'''
		returns weights, offset, scale
		'''
		return self.sess.run(self.weights), self.sess.run(self.offset), self.sess.run(self.scale)

	def score(self, X, y):
		'''
		likelihood of data. 
		'''
		likelihood = self.sess.run(self.log_loss, feed_dict = {self.design_:X, self.obs_: y_val})

		return likelihood


class exponential_GLM(GLM):
	'''
	Parameters:
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

	def __init__(self, weight_init,
		lr = 1e-2, eps = 1e-4, bias_init = 0.01, train_params = True,
		reg = 'l1', alpha = .1, non_lin = tf.sigmoid, scale_init = 1, verbose = True):
		'''
		Initialize the computational graph for the exponential GLM. Inherits from 
		the base GLM class, where all the instance variables are initialized / defined.
		
		'''	
		
		GLM.__init__(self, weight_init, lr, eps, bias_init, train_params,
			reg, alpha, non_lin, scale_init, verbose)

	def _logloss(self):
		'''

		'''	

		alpha = self.alpha

		
		
		fx = tf.matmul(self.design_, self.weights) - self.offset
		#fx = tf.reshape(fx, [-1, self.num_features, self.num_neurons])
		#fx = tf.reduce_sum(fx, reduction_indices = [1])- self.offset

		lam = self.non_lin(fx) - self.nonlin_offset

		self.scale = tf.nn.relu(self.scale)

		lam_ = tf.mul(self.scale,lam)+ self.eps
		

		coef = tf.log(lam_)
		distrib = tf.div(self.obs_, lam_)
		self.loss = tf.reduce_sum(coef + distrib, reduction_indices = [0])

		if self.reg == 'l2':	
			self.loss += alpha*tf.reduce_sum(tf.matmul(self.weights, self.weights, transpose_a = True))
			self.loss += alpha*tf.reduce_sum(tf.pow(self.scale, 2))
			self.loss += alpha*tf.reduce_sum(tf.pow(self.offset, 2))

		if self.reg == 'l1': 
			self.loss += alpha*tf.reduce_sum(self.weights + self.offset + self.scale )
		
		return self.loss



class poisson_GLM(GLM):
	def __init__(self, weight_init, 
		lr = 1e-2, eps = 1e-4, bias_init = 3, train_params = True,
		reg = 'l1', alpha = .1, non_lin = tf.sigmoid, scale_init = 1, verbose = True):
		'''
		initializes the computational graph for a poisson GLM with either exponential
		or sigmoidal non-linearity. 
		'''			

		GLM.__init__(self, weight_init, lr, eps, bias_init, train_params,
			reg, alpha, non_lin, scale_init, verbose)

	def _logloss(self):
		'''
		Poisson _logloss
		'''

		alpha = self.alpha

		fx = tf.matmul(self.design_, self.weights) -self.offset
		#fx = tf.reshape(fx, [-1, self.num_features, self.num_neurons])
		#fx = tf.reduce_sum(fx, reduction_indices =[1])- self.offset

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



class gaussian_GLM(GLM):
	def __init__(self, weight_init,
		lr = 1e-2, eps = 1e-4, bias_init = 0.01, train_params = True,
		reg = 'l1', alpha = .1, non_lin = tf.sigmoid, scale_init = 1, verbose = True):
		'''
		initializes the computational graph for a poisson GLM with either exponential
		or sigmoidal non-linearity. 
		'''	

		GLM.__init__(self, weight_init, lr, eps, bias_init, train_params,
			reg, alpha, non_lin, scale_init, verbose)

	def _logloss(self):
		'''
		Gaussian Log loss
		'''

		alpha = self.alpha

		fx = tf.matmul(self.design_, self.weights) - self.offset
		#fx = tf.reshape(fx, [-1, self.num_features, self.num_neurons])
		#fx = tf.reduce_sum(fx, reduction_indices = [1])- self.offset
		
		lam = self.non_lin(fx) - self.nonlin_offset

		self.scale = tf.nn.relu(self.scale)
		lam_ = tf.mul(self.scale,lam)+ self.eps
		
		#returns a separate loss for each neuron
		self.loss = tf.reduce_sum(tf.pow(self.obs_ - lam_, 2), reduction_indices = [0])

		if self.reg == 'l2':	
			self.loss += alpha*tf.reduce_sum(tf.matmul(self.weights, self.weights, transpose_a = True))
			self.loss += alpha*tf.reduce_sum(tf.pow(self.scale, 2))
			self.loss += alpha*tf.reduce_sum(tf.pow(self.offset, 2))

		if self.reg == 'l1': 
			self.loss += alpha*tf.reduce_sum(self.weights + self.offset + self.scale )
		
		return self.loss



class lognormal_GLM(GLM):
	def __init__(self, weight_init,
		lr = 1e-2, eps = 1e-4, bias_init = 0, train_params = True,
		reg = 'l1', alpha = .1, non_lin = tf.sigmoid, scale_init = 1, verbose = True):
		'''
		initializes the computational graph for a poisson GLM with either exponential
		or sigmoidal non-linearity. 
		'''	

		GLM.__init__(self, weight_init, lr, eps, bias_init, train_params,
			reg, alpha, non_lin, scale_init, verbose)

	def _logloss(self):
		'''
		Gaussian Log loss
		'''

		alpha = self.alpha

		fx = tf.matmul(self.design_, self.weights) - self.offset
		#fx = tf.reshape(fx, [-1, self.num_features, self.num_neurons])
		#fx = tf.reduce_sum(fx, reduction_indices = [1])- self.offset
		
		lam = self.non_lin(fx) 
		lam_ = tf.mul(self.scale,lam)+ self.eps
		
		#returns a separate loss for each neuron
		self.loss = tf.reduce_sum(tf.pow(tf.log(self.obs_) - lam_, 2), reduction_indices = [0])

		if self.reg == 'l2':	
			self.loss += alpha*tf.reduce_sum(tf.matmul(self.weights, self.weights, transpose_a = True))
			self.loss += alpha*tf.reduce_sum(tf.pow(self.scale, 2))
			self.loss += alpha*tf.reduce_sum(tf.pow(self.offset, 2))

		if self.reg == 'l1': 
			self.loss += alpha*tf.reduce_sum(self.weights + self.offset + self.scale )
		
		return self.loss