from glm_utils import *
import pickle
import itertools as it
import multiprocessing
import sys


def save_results(args):

	region, cre_line = args
	data_set = download_data([region], [cre_line])
	arranged_data = arrange_data_rs(data_set)
	results = make_tuning_curves(arranged_data)
	output = open('./boc/formatted/' + region + cre_line + 'rsresults' + '.pkl', 'wb')
	pickle.dump(results, output)
	output.close()

	return

if __name__ == '__main__':
	boc = BrainObservatoryCache(manifest_file='boc/manifest.json')
	regions = ['VISal']#, 'VISl', 'VISp', 'VISpm']
	lines =  ['Cux2-CreERT2', 'Rbp4-Cre', 'Rorb-IRES2-Cre'] 
    
	jobs = []
	for reg, cre in it.product(regions, lines):
		p = multiprocessing.Process(target=save_results, args=((reg, cre),))
		jobs.append(p)
		p.start()