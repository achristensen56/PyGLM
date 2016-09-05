from glm_utils import *
import group_glm as gm
from sklearn.cross_validation import train_test_split


def fit_glm(dff_array, design_matrix, non_linearity = [tf.exp, tf.nn.sigmoid, tf.nn.relu], noise_model = ['exponential', 'gaussian'] ):
'''
dff_array (numpy array) = n_time_bins x n_cells
design_matrx (numpy array) = n_time_bins n_features
non_linearity = list of non-linearities to fit
noise_models = list of noise models to fit

returns:
	scores = n_cells x n_non_linearities x n_noise_models
		
		likelihood on the test set for each non_linearity, and noise model 

	models = n_cells x n_non_linearities x n_noise_models x n_features
		
		the learned features -- average of the models learned over the 5 draws
'''


n_time_bins, n_cells = dff_array.shape
_, n_features = design_matrix.shape

n_nl, n_nm = len(non_linearity), len(noise_model)

scores = np.zeros([n_cells, n_nl, n_nm])
features = np.zeros([n_cells, n_nl, n_nm, n_features]) 

nm_dict = {'exponential': gm.exponential_GLM, 'gaussian': gm.gaussian_GLM}


for nm in noise_model:
	for nl in non_linearity:
		
		tp = True
		lr = 1e-3

		if nl == tf.exp or nl == tf.relu:
			#the exponential non-linearity 
			tp = False
			lr = 1e-5


		#here we fit 5 different versions of the model, on different test sets, and average them together
		test_l = []

		weights = []
		offset = []
		scale = []

		for i in range(5):

			train_test_data = train_test_split(design_matrix, dff_array)

			weight_init = np.linalg.pinv(X_train).dot(y_train)

			model = nm_dict[nm](weight_init, 
				lr = lr, train_params = tp, eps = 1e-4, bias_init = 0, alpha = 0, non_lin = nl, verbose = False)

			L, l = model.fit(*train_test_data)
			test_l.append(l[-1])

			w, o, s = model.get_params()	

		
			weights.append(w)
			offset.append(o)
			scale.append(s)

		#the parameters
		w = np.mean(weights)
		o = np.mean(offset)
		s = np.mean(offset)

		#the average of the final test likelihood is what we actually use for model comparison
		l = np.mean(test_l)
		






return scores, features	 