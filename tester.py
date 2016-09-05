from glm_utils import *

if __name__ == '__main__':
	regions = ['VISal', 'VISl', 'VISp', 'VISpm']
	lines =  ['Cux2-CreERT2', 'Rbp4-Cre', 'Rorb-IRES2-Cre'] 

	data_set = download_data(regions, lines)

	print data_set.keys()
