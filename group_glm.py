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
	#offset_init = 0.01*np.ones([num_neurons]), scale_init= 0.01*np.ones([num_neurons])

	def __init__(self, weight_init,
		lr = 1e-2, eps = 1e-4, bias_init = 0.01, train_params = True, offset_init = 0.01,
		reg = 'l1', alpha = .1, non_lin = 'sigmoid', scale_init = 0.01, verbose = True, cap_grads = True):
		'''
		Initialize the computational graph for the exponential GLM. 

		weight_init: self.num_features, self.num_neurons initialization for weights
		
		'''	



		if non_lin == 'sigmoid' :
			self.np_nonlin = sigmoid
			self.non_lin = tf.nn.sigmoid
		if non_lin == 'identity' :
			self.non_lin = tf_identity
			self.np_nonlin = tf_identity
		if non_lin == 'exp':
			self.np_nonlin = np.exp
			self.non_lin = tf.exp
		if non_lin == 'soft_rect':
			self.non_lin = tf_soft_rec
			self.np_nonlin = np_soft_rec

		self.num_features, self.num_neurons = weight_init.shape

		self.alpha, self.reg, self.verbose =  alpha, reg, verbose
		self.lr, self.eps = lr, eps


		self.design_ = tf.placeholder('float32', [None, self.num_features])
		self.obs_ = tf.placeholder('float32', [None, self.num_neurons])
		
		self.weights = tf.Variable(weight_init, dtype = 'float32', name = 'weights')

		self.offset = tf.Variable(bias_init, dtype = 'float32', trainable = train_params)
		self.scale = tf.Variable(scale_init, dtype = 'float32', trainable = train_params)
		self.nonlin_offset = tf.Variable(offset_init, dtype = 'float32')

		self.log_loss = self._logloss()

		if cap_grads == True:	
			optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
			
			#optimizer = tf.train.GradientDescentOptimizer(learning_rate = self.lr)
			self.gvs = optimizer.compute_gradients(self.log_loss)
			capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in self.gvs]

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

		for i in range(max_iters):
		    idx = np.random.randint(0, T, size = batch_size)
		    
		    train_feat = X[idx] 
		    train_obs = y[idx]

		    if self.verbose:
		    	bar.update()
		    
		    l, _ = self.sess.run([self.log_loss, self.train_step], feed_dict = {self.design_:train_feat, self.obs_: train_obs})
			 

		    #print "the loss is: ", l, " it's shape is: ", l.shape
		    #print "the current offset value is: ", off, "it's shape is: ", off.shape
		    #print "the current input to the non-linearity is: " , fx, " it's shape is: ", fx.shape
		    #print "the current output of the non-linearity is: ", lam_, " it's shape is: ", lam_.shape
		    #print "the current distrib is: ", distrib, " it's shape is: ", distrib.shape
		    #print "the current coef is: ", coef, " it's shape is: ", coef.shape

		    if i % 10 == 0 and X_val != None:

		    	l1 = self.sess.run(self.log_loss, feed_dict = {self.design_:X_val, self.obs_:y_val})
		    	cross_val.append(l1)
		    	    
		    loss_arr.append(l)	    

		return np.array(loss_arr), np.array(cross_val) 


	def predict_trace(self, X):
		return self.sess.run(self.predict, feed_dict = {self.design_:X})

	def get_params(self):
		'''
		returns weights, offset, scale
		'''
		
		return self.sess.run(self.weights), self.sess.run(self.offset), self.sess.run(self.nonlin_offset), self.sess.run(self.scale)

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
		lr = 1e-2, eps = 1e-4, bias_init = 0.01, train_params = True, offset_init = 0.01, cap_grads = True,
		reg = 'l1', alpha = .1, non_lin = tf.sigmoid, scale_init = 0.01, verbose = True):
		'''
		Initialize the computational graph for the exponential GLM. Inherits from 
		the base GLM class, where all the instance variables are initialized / defined.
		
		'''	
		tf.reset_default_graph()
		self.sess = tf.Session()
		
		#just for gamma GLM

		
		GLM.__init__(self, weight_init, lr, eps, bias_init, train_params, offset_init,
			reg, alpha, non_lin, scale_init, verbose, cap_grads)

	def _logloss(self):
		'''

		'''	

		alpha = self.alpha

		
		
		self.fx = tf.matmul(self.design_, self.weights) - self.offset
		#fx = tf.reshape(fx, [-1, self.num_features, self.num_neurons])
		
		#self.fx = tf.reduce_sum(self.fx, reduction_indices = [0])

		self.nonlin_offset = tf.nn.relu(self.nonlin_offset) + self.eps
		self.scale = tf.nn.relu(self.scale) + self.eps

		self.lam = self.non_lin(self.fx) + self.nonlin_offset
		#self.lam = tf.nn.relu(self.lam) + self.eps
		self.lam_ = tf.mul(self.scale,self.lam)+ self.eps


		self.coef = tf.log(self.lam_)
		self.distrib = tf.div(self.obs_, self.lam_)
		self.loss = tf.reduce_sum(self.coef + self.distrib, reduction_indices = [0])

		if self.reg == 'l2':	
			self.loss += alpha*tf.reduce_sum(tf.matmul(self.weights, self.weights, transpose_a = True))
			self.loss += alpha*tf.reduce_sum(tf.pow(self.scale, 2))
			self.loss += alpha*tf.reduce_sum(tf.pow(self.offset, 2))

		if self.reg == 'l1': 
			self.loss += alpha*tf.reduce_sum(self.weights + self.offset + self.scale )
		
		return self.loss


	def get_log_likelihood(self, data, stimulus):
		A = self.sess.run(self.log_loss, feed_dict = {self.design_: stimulus, self.obs_: data})
		return -A *  1./len(data)

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
		



class gaussian_GLM(GLM):
	def __init__(self, weight_init,
		lr = 1e-2, eps = 1e-4, bias_init = 0.01, train_params = True, offset_init = 0.01,
		reg = 'l1', alpha = .1, non_lin = tf.sigmoid, scale_init = 0.01, verbose = True):
		'''
		initializes the computational graph for a poisson GLM with either exponential
		or sigmoidal non-linearity. 
		'''	
		tf.reset_default_graph()
		self.sess = tf.Session()
		
		#just for gamma GLM

		GLM.__init__(self, weight_init, lr, eps, bias_init, train_params, offset_init, 
			reg, alpha, non_lin, scale_init, verbose)

	def _logloss(self):
		'''
		Gaussian Log loss
		'''

		alpha = self.alpha

		fx = tf.matmul(self.design_, self.weights) - self.offset
		#fx = tf.reshape(fx, [-1, self.num_features, self.num_neurons])
		#fx = tf.reduce_sum(fx, reduction_indices = [1])- self.offset
		

		self.nonlin_offset = tf.nn.relu(self.nonlin_offset) + self.eps

		lam = self.non_lin(fx) + self.nonlin_offset

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

	def get_log_likelihood(self, data, stimulus):
		A = self.sess.run(self.log_loss, feed_dict = {self.design_: stimulus, self.obs_: data})
		N = float(len(data))
		w, o, nls, s = self.get_params()
		var = gaussian_variance(data, stimulus, w, o, nls, s, non_lin = self.np_nonlin)
		#var = 1.12
		l = 1/N * (-N/2 * np.log(2*np.pi) - N/2 * np.log(var) - 1 / (2*var) * A)
		return l

class gamma_GLM(GLM):
	def __init__(self, weight_init,
		lr = 1e-2, eps = 1e-4, bias_init = 0.01, train_params = True, offset_init = 0.01, cap_grads = True,
		reg = 'l1', alpha = .1, non_lin = tf.sigmoid, scale_init = 0.01, verbose = True, alpha_init = 1):
		'''
		Initialize the computational graph for the gamma GLM. Inherits from 
		the base GLM class, where all the instance variables are initialized / defined.
		
		'''	
		tf.reset_default_graph()
		self.sess = tf.Session()
		self.num_features, self.num_neurons = weight_init.shape
		
		#just for gamma GLM
		self.alpha_param = 1.5*tf.Variable(np.ones([1, self.num_neurons]), dtype = 'float32', trainable = True)
		
		GLM.__init__(self, weight_init, lr, eps, bias_init, train_params, offset_init,
			reg, alpha, non_lin, scale_init, verbose, cap_grads)

		

	def _logloss(self):
		'''

		'''	

		alpha = self.alpha

		
		
		self.fx = tf.matmul(self.design_, self.weights) - self.offset
		#fx = tf.reshape(fx, [-1, self.num_features, self.num_neurons])
		
		#self.fx = tf.reduce_sum(self.fx, reduction_indices = [0])

		self.nonlin_offset = tf.nn.relu(self.nonlin_offset) + self.eps
		self.scale = tf.nn.relu(self.scale) + self.eps
		self.alpha_param  = tf.nn.relu(self.alpha_param) + self.eps 

		self.lam = self.non_lin(self.fx) + self.nonlin_offset
		#self.lam = tf.nn.relu(self.lam) + self.eps
		self.lam_ = tf.mul(self.scale,self.lam)+ self.eps
		self.lam_ = tf.div(self.alpha_param, self.lam_)

		

		self.loss =  -tf.reduce_sum(tf.contrib.distributions.Gamma(self.alpha_param, self.lam_).log_pdf(self.obs_), reduction_indices = [0])
		

		#self.coef = tf.log(self.lam_)
		#self.distrib = tf.div(self.obs_, self.lam_)
		#self.loss = tf.reduce_sum(self.coef + self.distrib, reduction_indices = [0])

		if self.reg == 'l2':	
			self.loss += alpha*tf.reduce_sum(tf.matmul(self.weights, self.weights, transpose_a = True))
			self.loss += alpha*tf.reduce_sum(tf.pow(self.scale, 2))
			self.loss += alpha*tf.reduce_sum(tf.pow(self.offset, 2))

		if self.reg == 'l1': 
			self.loss += alpha*tf.reduce_sum(self.weights + self.offset + self.scale )
		
		return self.loss	

	def get_log_likelihood(self, data, stimulus):
		A = self.sess.run(self.log_loss, feed_dict = {self.design_: stimulus, self.obs_: data})
		return -A *  1./len(data)



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