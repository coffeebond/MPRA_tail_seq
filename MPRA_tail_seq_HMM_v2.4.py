'''
It calculates poly(A) tail length based on a Gaussian mixture hidden markov model trained on a subset of the data
It outputs these files:
1. "*_log.txt", a log file for the run
2. "*_pA_less_all_tags.txt", if the 3' adapter is checked, this file is written containing those reads with 3' adapters found
	before the supposed poly(A) start position, with the following infomation:
	1) id
	2) sequence from the random region (maybe partial if the 3' adapter is found within the random region)
	3) lowest QC for the random region 
	4) lowest hamming distance for the constant region (NC: not changed; NF: not found)
	5) position offset of the random region determined by the constant region
	6) lowest hamming distance between the found 3' adapter and the reference adapter (NC: not changed; NF: not found)
	7) the position index of the found 3' adapter (NC: not changed; NF: not found)
	8) tail length  
3. "_pA_less_stat.txt", if the 3' adapter is checked, this file is written with tail length statistics calculated
	from "*_pA_less_all_tags.txt"
4. "*_pA_all_tags.txt", all clusters (tags) containing: 
	1) id
	2) sequence from the random region (maybe partial if the 3' adapter is found within the random region)
	3) lowest QC for the random region 
	4) lowest hamming distance for the constant region (NC: not changed; NF: not found)
	5) position offset of the random region determined by the constant region
	6) lowest hamming distance between the found 3' adapter and the reference adapter (NC: not changed; NF: not found)
	7) the position index of the found 3' adapter (NC: not changed; NF: not found)
	8) tail length  
5. "_pA_stat.txt", tail length statistics calculated from "*_pA_all_tags.txt", containing: 
	1) sequence from the random region
	2) median tail length
	3) number of duplicates
	4) tail length at quantile 10, 25, 75, 90
6. "_states.txt", a file (to be zipped later) containing:
	1) id
	2) sequence from the random region
	3) all HMM states
7. "_HMM_model.txt", the HMM model 
8. A temporary file containing converted T-signal will be written out on the disk to save memory usage
	and it can be deleted in the end. (only in mode 1)

###--- v2.1 change log:
	1. Added an alternative mode used to call poly(A) tail lengths, after seeing many clusters wrongly called 0 tail lengths
	2. Added a section in which pA-less reads are aggregated by the random region and written out in a separate file

###--- v2.2 change log:
	1. Corrected the format of the output file for the pA-less reads (determined by "check_3_adapter").
	2. added a log file which records all print output
	3. added a parser for parsing the input argument, with options to control some preset parameters
	4. added an option to check if a read is a spike-in 
	5. changed the criteria to filter a read that has no match to the constant region: only if this is not caused by the 3' adapter found 
		before the poly(A) start position
	6. changed the output values for functions "check_spike_in", "check_constant_region" and "check_3_adapter"
	7. moved the HHM states output to a separate file, which can be deleted later to save disk space
	8. all tail-less single tags as well as their aggregated stats will be written into two separate files

###--- v2.3 change log (20210519):
	1. changed the way to call poly(A) tail lengths 
	2. changed the definition of the parameter "min_match" for checking 3' adapters at the very 3' ends to make it on par with the allowed 
		error rate when the adapter appear in the middle of the read

###--- v2.4 change log (20240917):
	1. changed global variables to an ordered dictionary
	2. changed the way to define spike-in, making it compatible with pA standards
'''


import sys, subprocess, math, gzip, ghmm, time, tarfile, concurrent.futures, random, os, argparse
import numpy as np
from ghmm import *
from time import time
from datetime import datetime
from collections import OrderedDict

###------global variables-----------------------------------------------------------------

params = OrderedDict([
	# !!! The following parameters need to be set to ensure a correct mode of this script !!! ----------
	
	# 1. Whether to use QC score in fastq file to filter reads.
	('flag_filter_by_qc', True), 
	('qc_code', '@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefgh'), #QC score coding, default is Illumina 1.5
	('qc_cutoff', 10), # minimal QC score for bases in the random region
	
	# 2. Whether to use a constant region between random region and poly(A) region to 
	#    ensure quality and find the right position of the start of poly(A) region
	('flag_check_constant_region', True), #This can be changed by the input argument (c).
	('c_seq', 'AATAAAGAAATTGATTTGTCT'), # a constant region before poly(A) region for quality control
	('c_seq_pos', 61), # start position of the constant sequence region
	('c_hmdis_cutoff', 6), # hamming distance between the read and the constant sequence must be no more than this  
	('c_wig', 3), # number of bases upstream or downstream of the expected position of the constant region
	
	# 3. Whether to check the 3' adapter
	('flag_check_3_adapter', True), # This can be changed by the input argument (a).
	('a_seq', 'TCGTATGCCGTCTTCTGCTTG'), # 3' adapter sequence
	('error_rate', 0.2), # percentage of bases allowed to have mismatches (floor integer, int(21 * 0.2) = 4)
	('min_match', 16), # minimal number of matches to consider when the adapter is examined at the very 3' end
	('min_r_seq_len', 6), # minimal length of the random sequence, left after the 3' adapter is found within the random region and trimmed

	# 4. Whether to check 'N' base
	('flag_check_base_N', True),

	# 5. Whether to check spike-in 
	('flag_check_spike_in', True),
	('sp_seq', 'ACCAGCCTCAAGAACACCCGAATGG'),
	('sp_start', 1),
	('sp_wig', 3), # number of bases upstream or downstream of the expected position of the spike-in, cannot be smaller than 'sp_start'
	('sp_hmdis_cutoff', 5),

	# 6. What HMM mode to use
	('allow_back', True), # whether to allow bi-directional state transition
	('mixed_model', True), # whether to use mixed gaussian model for the input variable

	# The following parameters determines positions of certain features in the fastq file
	('n_start', 1), # start position of the random region
	('n_end', 60), # end position of the random region
	('pa_start', 82), # start position of the poly(A) region. This can be changed by the input argument (p).

	# The following parameters are for HMM training
	('training_max', 50000), # maximal number of clusters used in the training set
	('training_min', 5000), # minimal number of clusters used in the training set
	('bound', 5), # boundary for normalized log2 A_signal
	('all_zero_limit', 50), # limit for total number of all zeros in four channels in the intensity file

	# The following parameters are for tail-length calling
	('flag_pa_override', True), # whether to override tail length by using position indice of the 3' adapter and the constant region
	('non_A_allowed', 4), # maximal number of non-A states at the beginning of the poly(A) region 
	# sometimes A-signal can be unexpcted low at the beginning of the poly(A) region, it's not clear why
	# The value is set at 4 because the adapter sequence has A at the fifth position

	# The following parameters are for multiprocessing
	('n_threads', 20), # number of cores to use for multiprocess, if too big, memory may fail
	('chunk_lines', 12500), # number of lines to allocate to each core to process, if too big, memory may fail
	('chunk', 1000000) # give a feedback for proceessing this number of lines
])
# update the 'min_match' value
params['min_match'] = len(params['a_seq']) - int(len(params['a_seq']) * params['error_rate'])

'''
# !!! The following parameters need to be set to ensure a correct mode of this script !!! ----------
# 1. Whether to use QC score in fastq file to filter reads.
flag_filter_by_qc = True
qc_code = '@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefgh' #QC score coding, default is Illumina 1.5
qc_cutoff = 10 # minimal QC score for bases in the random region

# 2. Whether to use a constant region between random region and poly(A) region to 
#	ensure quality and find the right position of the start of poly(A) region
flag_check_constant_region = True # This can be changed by the input argument (c).
c_seq = 'AATAAAGAAATTGATTTGTCT' # a constant region before poly(A) region for quality control
c_seq_pos = 61 # start position of the constant sequence region
c_hmdis_cutoff = 6 # hamming distance between the read and the constant sequence must be no more than this  
c_wig = 3 # number of bases upstream or downstream of the expected position of the constant region

# 3. Whether to check the 3' adapter
flag_check_3_adapter = True # This can be changed by the input argument (a).
a_seq = 'TCGTATGCCGTCTTCTGCTTG' # 3' adapter sequence
error_rate = 0.2 # percentage of bases allowed to have mismatches (floor integer, int(21 * 0.2) = 4)
min_match = len(a_seq) - int(len(a_seq) * error_rate) # minimal number of matches to consider when the adapter is examined at the very 3' end
min_r_seq_len = 6 # minimal length of the random sequence, left after the 3' adapter is found within the random region and trimmed

# 4. Whether to check 'N' base
flag_check_base_N = True

# 5. Whether to check spike-in 
flag_check_spike_in = False
sp_seq = 'ACCAGCCTCAAGAACACCCGAATGG'
sp_start = 1 
sp_wig = 3 # number of bases upstream or downstream of the expected position of the spike-in, cannot be smaller than 'sp_start'
sp_hmdis_cutoff = 5

# 6. What HMM mode to use
allow_back = True # whether to allow bi-directional state transition
mixed_model = True # whether to use mixed gaussian model for the input variable

# The following parameters determines positions of certain features in the fastq file
n_start = 1 # start position of the random region
n_end = 60 # end position of the random region
pa_start = 61 # start position of the poly(A) region. This can be changed by the input argument (p).

# The following parameters are for HMM training
training_max = 50000 # maximal number of clusters used in the training set
training_min = 5000 # minimal number of clusters used in the training set
bound = 5 # boundary for normalized log2 A_signal
all_zero_limit = 50 # limit for total number of all zeros in four channels in the intensity file

# The following parameters are for tail-length calling
flag_pa_override = True # whether to override tail length by using position indice of the 3' adapter and the constant region
non_A_allowed = 4 # maximal number of non-A states at the beginning of the poly(A) region 
# sometimes A-signal can be unexpcted low at the beginning of the poly(A) region, it's not clear why
# The value is set at 4 because the adapter sequence has A at the fifth position

# The following parameters are for multiprocessing
n_threads = 20 # number of cores to use for multiprocess, if too big, memory may fail
chunk_lines = 12500 # number of lines to allocate to each core to process, if too big, memory may fail
chunk = 1000000 # give a feedback for proceessing this number of lines
'''
###------functions------------------------------------------------------------------------
def fread(file): # flexibly read in files for processing
	file = str(file)
	if file.endswith('tar.gz'):
		temp = tarfile.open(file, 'r:gz')
		return temp.extractfile(temp.next())
	elif file.endswith('.gz'):
		return gzip.open(file, 'rt')
	elif file.endswith('.txt'):
		return open(file, 'r')
	else:
		sys.exit("Wrong file type to read: " + file)

def hamming_distance(s1, s2):
    return sum(ch1 != ch2 for ch1,ch2 in zip(s1,s2))

def check_spike_in(f_seq):
	# checks if the read is a spike-in, which won't be examined for tail length
	# returns one the following:
	# 1) 'NC', if it's not checked
	# 2) 'NS', if it's checked but it's not a spike-in
	# 3) hm_min, if it's checked and it's a spike-in
	if params['flag_check_spike_in']:
		seq_lst = [f_seq[max(params['sp_start']-1+i, 0):min(params['sp_start']-1+len(params['sp_seq'])+i, len(params['sp_seq']))] for i in range(-params['sp_wig'],params['sp_wig']+1)]
		hm_lst = [hamming_distance(seq_lst[i], params['sp_seq']) for i in range(len(seq_lst))]
		hm_min = min(hm_lst)
		if hm_min <= params['sp_hmdis_cutoff']: # hamming distance must be no larger than this
			return(hm_min)
		else:
			return('NS')
	else:
		return('NC')

def check_constant_region(f_seq):
	# use a constant region (if exists) to calibrate the start of the poly(A) region
	# returns one of the following:
	#	1) if it is checked and hm_min is no larger than the cutoff:
	#		(lowest hamming distance, position offset)
	# 	2) if it is checked but hm_min is larger than the cutoff:
	#		('c_NM', 0)
	#	3) if it is not checked:
	#		('c_NC', 0)
	if params['flag_check_constant_region']:
		seq_lst = [f_seq[(params['c_seq_pos']-1+i):(params['c_seq_pos']-1+len(params['c_seq'])+i)] for i in range(-params['c_wig'],params['c_wig']+1)]
		hm_lst = [hamming_distance(seq_lst[i], params['c_seq']) for i in range(len(seq_lst))]
		hm_min = min(hm_lst)
		if hm_min <= params['c_hmdis_cutoff']: # hamming distance must be no larger than this
			pos_offset = hm_lst.index(hm_min) - params['c_wig']
		else:
			hm_min = 'c_NM'
			pos_offset = 0
	else:
		hm_min = 'c_NC'
		pos_offset = 0
	return(hm_min, pos_offset)

def check_3_adapter(f_seq, pos_offset):
	# find the 3' adapter postion in the read (if one can be found)
	# returns one of the following: 
	#	1) if it is checked and the length of the resulting random region (after trimming) is no smaller than min_r_seq_len: 
	#		(the hamming distance between the found adapter and the reference adapter, the index of the 3' adapter position) 
	# 	2) if it is checked and the length of the resulting random region (after trimming) is smaller than min_r_seq_len:
	#		None
	# 	3) if it is checked but not found:
	#		('a_NF_hmd', 'a_NF_pos')
	#	4) if it is not checked
	#		('a_NC_hmd', 'a_NC_pos')
	if params['flag_check_3_adapter']:
		# first check if there is a match in the body, i.e. not the very 3' end
		seq_lst = [f_seq[i:i+len(params['a_seq'])] for i in range(len(f_seq)-len(params['a_seq'])+1)]
		mm_lst = np.asarray([hamming_distance(seq_lst[i], params['a_seq']) for i in range(len(seq_lst))])
		hm_min = min(mm_lst)
		if hm_min <= int(len(params['a_seq']) * params['error_rate']):
			idx_lst = np.where(mm_lst == min(mm_lst))[0]
			if len(idx_lst) == 1:
				idx_a_seq = idx_lst[0]
			elif all(idx_lst < params['pa_start'] + pos_offset - 1):
				idx_a_seq = idx_lst[-1]
			else:
				idx_a_seq = [x for x in idx_lst if x >= params['pa_start'] + pos_offset - 1][0]
			if idx_a_seq < max(0, params['n_start'] - 1 + pos_offset) + params['min_r_seq_len']:
				return(None)
		# then check if there is a match at the very 3' end, with a minimal length overlapping with the adapter
		else:
			end_seq_lst = [f_seq[len(f_seq)-len(params['a_seq'])+1+i:len(f_seq)] for i in range(len(params['a_seq'])-1-(params['min_match']-1))]
			a_seq_lst = [params['a_seq'][:j] for j in map(len, end_seq_lst)]
			mm_lst = [hamming_distance(end_seq_lst[i], a_seq_lst[i]) for i in range(len(end_seq_lst))]
			idx_lst = [i for i in range(len(mm_lst)) if  mm_lst[i] <= int(len(end_seq_lst[i]) * params['error_rate'])]
			if len(idx_lst) != 0:
				idx_a_seq = len(f_seq)-len(params['a_seq']) + 1 + idx_lst[0]
				hm_min = mm_lst[idx_lst[0]]
			else:
				# if an adapter sequence match can't be found, return this 
				hm_min= 'a_NF_hmd'
				idx_a_seq = 'a_NF_pos'
		return(hm_min, idx_a_seq)
	else:
		# return 'a_NC_hmd','a_NC_pos' if this step is not performed
		return('a_NC_hmd','a_NC_pos')

def chekc_base_N(r_seq):
	if params['flag_check_base_N'] and 'N' in r_seq:
		return(None)
	else:
		return(r_seq)

def filter_qc(qc_str):
	qc_min = min([params['qc_code'].index(x) for x in qc_str])
	if params['flag_filter_by_qc'] and qc_min <= params['qc_cutoff']:
		return(None)
	else:
		return(qc_min)

def worker_fq(lst): # input: [[line1, line2, line4], ... ]
	# the order to quality control
	#	1) check spike-in
	#	2) check constant region 
	#	3) check 3' adapter
	#	4) check N in the random region
	#	5) check QC in the random region
	for_hmm_dict = {} # updated library 
	no_hmm_dict = {}
	lost_ary = np.zeros(4)
	sp_count_ary = np.zeros(params['sp_hmdis_cutoff']+1)
	for sub_lst in lst:
		f_seq = sub_lst[1]
		sp_res = check_spike_in(f_seq)
		if isinstance(sp_res, str):
			res_lst = check_constant_region(f_seq)
			hm_min_c = res_lst[0]
			pos_offset = res_lst[1]
			a_lst = check_3_adapter(f_seq, pos_offset)
			if a_lst != None:
				hm_min_a = a_lst[0]
				idx_a_seq = a_lst[1]
				if isinstance(idx_a_seq, str) == False and idx_a_seq < params['n_end'] + pos_offset:
					r_seq = f_seq[max(0, (params['n_start'] - 1 + pos_offset)):idx_a_seq]
					qc = sub_lst[2][max(0, (params['n_start'] - 1 + pos_offset)):idx_a_seq]
				else:
					r_seq = f_seq[max(0, (params['n_start'] - 1 + pos_offset)):(params['n_end'] + pos_offset)]
					qc = sub_lst[2][max(0, (params['n_start'] - 1 + pos_offset)):(params['n_end'] + pos_offset)]
				res = chekc_base_N(r_seq)
				if res != None:
					r_seq = res
					qc_min = filter_qc(qc)
					if qc_min != None:
						idx = ':'.join(sub_lst[0].split('#')[0].split(':')[-3:])
						if isinstance(idx_a_seq, str) == False and idx_a_seq < params['pa_start'] - 1 + pos_offset:
							# this read has no poly(A) tail and won't be examined in HMM
							no_hmm_dict.setdefault(idx, [r_seq, qc_min, hm_min_c, pos_offset, hm_min_a, idx_a_seq])
						else:
							if hm_min_c == 'c_NM':
								lost_ary[0] += 1 # number of reads that have no match to the constant sequence region (and not because of the 3' adapter)
							else:
								for_hmm_dict.setdefault(idx, [r_seq, qc_min, hm_min_c, pos_offset, hm_min_a, idx_a_seq])
					else:
						lost_ary[3] += 1 # number of reads that have bases with too low QC scores
				else:
					lost_ary[2] += 1 # number of reads that have 'N' base in the random region
			else:
				lost_ary[1] += 1 # number of reads that has a 3' adapter sequence located in the random region, making the random sequence too short if the 3' adapter is trimmed
		else:
			sp_count_ary[sp_res] += 1 # number of reads that match the spike-in sequence
	return(for_hmm_dict, no_hmm_dict, lost_ary, sp_count_ary)

	
def Convert2A(line):
# this function does two things:
# 1. normalize intensity for all four channels of each cluster, using the random region 
	dict_value = {'A':[] ,'C':[] ,'G':[], 'T':[]}
	lst = line.rstrip().split('\t') 
	idx = ':'.join(lst[:3])
	pos_offset = mdict[idx][3]
	for i in range(max(0, (params['n_start'] - 1 + pos_offset)), params['n_end'] + pos_offset): # i is the index of the i+1 base in the random region 
		lst4c = map(int, [x for x in lst[i+4].split(' ') if x != '']) 
		# first 4 elements are not intensities
		# get rid of spaces between intensity values
		dict_4c = {'A':lst4c[0] ,'C':lst4c[1] ,'G':lst4c[2], 'T':lst4c[3]}
		base = mdict[idx][0][i - max(0, (params['n_start'] - 1 + pos_offset))] # the index in the intensity line is different from the index in the random sequence in the dictionary
		dict_value[base].append(dict_4c[base]) 
	for key in dict_value:
		if len(dict_value[key]) == 0 or np.mean(dict_value[key]) <= 0:
			return None
			# exit this function if normalization can't be completed
		else:
			dict_value[key] = np.mean(dict_value[key])
			# otherwise, take the average value
			# this should be the approximate value illumina used to call the base for that cluster
# 2. convert intensity from 4 channels to A signal
	all_A = []
	pa_start_new = params['pa_start'] + pos_offset + 1 # in the run performed on 20240828, cycle 26 was repeated as cycle 27, so add 1 cycle here as correction
	for j in range(pa_start_new+4-1, len(lst)): # j is the index of j+1 position in intensity line
		lst4c = map(int, [x for x in lst[j].split(' ') if x != ''])
		if lst4c == [0,0,0,0]:
			all_A.append('empty')
		# sometimes in a base position, all channel signals equal to 0
		# these need to be corrected later or discarded if there are too many in a cluster
		else:
			dict_4c = {'A':lst4c[0] ,'C':lst4c[1] ,'G':lst4c[2], 'T':lst4c[3]}
			for key in dict_4c:
				if dict_4c[key] <= 0:
					dict_4c[key] = 1.0 / dict_value[key]
				else:
					dict_4c[key] = float(dict_4c[key]) / dict_value[key]
				# normalize the intensity value 
			A_signal = dict_4c['A'] / (dict_4c['T'] + dict_4c['C'] + dict_4c['G'])
			A_signal = math.log(A_signal, 2)
			A_signal = max(-params['bound'], min(A_signal, params['bound']))
			# make large or small A_signal bound
			all_A.append(A_signal)
	if all_A.count('empty') >= params['all_zero_limit']:
		return None
	else:
		for k in range(len(all_A)):
			if all_A[k] == 'empty':
				sliding_A = all_A[max(0, k - params['all_zero_limit']):min(len(all_A), k + params['all_zero_limit'])]
				sliding_A = [x for x in sliding_A if x != 'empty']
				all_A[k] = np.mean(sliding_A)
		# if there more than params['all_zero_limit'] base positions with all channel signals being equal to 0, discard this cluster
		# else, use the mean in a sliding window to fill in the missing value
	all_A.insert(0, params['bound']*100) # add a peudo A to the front, making sure HMM starts with A
	all_A.extend([-params['bound']] * (pos_offset + params['c_wig'])) # pad the A-signal list due to wiggling start position
	return all_A

def worker_C2A(lines):
# this function takes all lines from a intensity file and allocates them to each process
# and outputs a tupple including
# 1. a new dictionary which contains gene_name and converted T-signal
# 2. a list containing converted A-signals to be trained
	temp_dict = {}
	temp_lst = []
	for line in lines:	
		idx = ':'.join(line.rstrip().split('\t')[:3])
		if mdict.has_key(idx):
			temp = Convert2A(line)
			if temp: # only reads that have converted T-signal will be used later			
				temp_dict.setdefault(idx, (mdict[idx] + [temp]))
				if idx in train_keys:
					temp_lst.append(temp)
	return temp_dict, temp_lst

def worker_hmm(lines):
# this function takes a number of lines containing converted A-signal and calculates the tail length 
# It returns:
# 1. a dictionary with id as the key, and the value containing the orignal line content except for the first column (key) and the 
#	last column (A-signal), and tail length calculated from HMM and final tail length (maybe overriden by adapter positions)
# 2. a list containing the random sequence and the final tail-length 
	temp_dict = {}
	temp_lst = []
	for line in lines:
		l = line.rstrip().split('\t')
		states = model.viterbi(EmissionSequence(F, map(float,l[len_mdict_value+1:])))[0]
		# tail length defined by the position followed by two non-A bases (state 3 and 4)
		# Also, the first position is a peudo-T base
		if len(states) == 0:
			sys.exit("Can't infer the states from the model! Exiting...")

		# convert the sequence to A (+1) or non-A (-1) state and pad the non A states in the end
		a_lst = [1 if x <= ((len(pi) - 1) / 2) else -1 for x in states] + [-1, -1] 

		# calculate the tail length
		flag_start = False
		for i in range(1, len(a_lst)-1):
			if flag_start == False:
				if a_lst[i] + a_lst[i+1] > 0: # need two A stats to start counting tail length
					tl_start = i
					flag_start = True
				else:
					if i > params['non_A_allowed']: # tail can't start beyond this region
						if 1 in a_lst[1:i]:
							tl_hmm = 1
						else:	
							tl_hmm = 0 
						break
			else:
				if a_lst[i] + a_lst[i+1] < 0: # need two non-A stats to stop counting tail length
					tl_end = i
					tl_hmm = tl_end - tl_start
					break

		temp_dict.setdefault(l[0], list(map(str,l[1:len_mdict_value+1]))+[str(tl_hmm), str(states)])
		temp_lst.append([l[1], tl_hmm])
	return temp_dict, temp_lst

def lines_sampler(filename, n_lines_sample):
	# this function takes in the reads_wTsignal file and
	# randomly select n_lines_sample lines to output as a list of lists (each containing T signals)
	sample = []
	with fread(filename) as f:
		f.seek(0, 2)
		filesize = f.tell()
		random_set = sorted(random.sample(xrange(filesize), n_lines_sample))
		for i in range(n_lines_sample):
			f.seek(random_set[i])
			
			# Skip current line (because we might be in the middle of a line) 
			f.readline()
			
			# Append the next line to the sample set 
			line = f.readline()
			if line:
				sample.append(list(map(float, line.rstrip().split('\t')[len_mdict_value+1:])))
	return sample
	
def timer(): # calculate runtime
	temp = str(time()-t_start).split('.')[0]
	temp =  '\t' + temp + 's passed...' + '\t' + str(datetime.now())
	return temp

def pwrite(f, text): # a decorator for print; it prints and also writes the content to a log file
	f.write(text + '\n')
	print(text)

def make_log_file(filename, p_params = False, p_vars = False):
	f_log = open(filename, 'w')
	if isinstance(p_params, dict):
		pwrite(f_log, 'Global parameters:')
		for param in p_params:
			pwrite(f_log, param + ': ' + str(p_params[param]))
	if isinstance(p_vars, dict):
		pwrite(f_log, '\nInput arguments:')
		for var in p_vars:
			pwrite(f_log, var + ': ' + str(p_vars[var]))
		pwrite(f_log, '\n')	
	return(f_log)

#####################################################################################################################
###------the script runs from there-------------------------------------------------------
t_start = time() # timer start

# parse the input
parser = argparse.ArgumentParser()
parser.add_argument('-o', '--out', dest = 'o', type = str, help = 'output file prefix')
parser.add_argument('-q', '--fastq', dest = 'q', type = str, help = 'input fastq file')
parser.add_argument('-i', '--intensity', dest = 'i', type = str, help = 'input intensity file')
parser.add_argument('-s', '--signal', dest = 's', type = str, help = 'input A signal file')
parser.add_argument('-c', '--constant_region', dest = 'c', action = 'store_true', default = False, help = 'flag for checking the constant region')
parser.add_argument('-a', '--adapter', dest = 'a', action = 'store_true', default = False, help = 'flag for checking the 3\' adapter region')
parser.add_argument('-p', '--pa_start', dest = 'p', type = int, default = params['pa_start'], help = 'starting position for poly(A) region')
parser.add_argument('-sp', '--spike_in', dest = 'sp', action = 'store_true', default = False, help = 'flag for checking spike_in')
args = parser.parse_args()  

# check input files and determine which mode to run
if args.s:
	hmm_only = True
else:
	if not args.q:
		sys.exit('Missing the input fastq file!')
	if not args.i:
		sys.exit('Missing the input intensity file!')
	hmm_only = False

###-------------------------------------------------	
if not hmm_only:  
		
	# read in all files and open files for writing
	
	# prefix for the output files
	if args.o:
		out_prefix = args.o
	else:
		out_prefix = './' + f_fastq.split('/')[-1].split('-s')[0] + f_fastq.split('/')[-1].split('-s')[1].split('_sequence')[0] 
	
	f_fastq = args.q # fastq file (un-trimmed)
	f_intensity = args.i # intensity file 
	out_A_signal = out_prefix + '_A_signal.txt' # output file for converted A signals
	out_states = out_prefix + '_states.txt' # output file for HMM states
	out_hmm = out_prefix + '_HMM_model.txt' # output file for HMM model
	out_single_tags = out_prefix + '_pA_all_tags.txt' # output file for all tags with poly(A) tails (by HMM)
	out_pa_stat = out_prefix + '_pA_stat.txt' # output file for poly(A) tail length stats for each unique random sequence
	out_tl_less_tags = out_prefix + '_pA_less_all_tags.txt' # output file for all tail-less tags 
	out_tl_less_stat = out_prefix + '_pA_less_stat.txt' # output file for stats for each unique random sequence
	out_log = out_prefix + '_log.txt' # log file output
	
	if args.c:
		params['flag_check_constant_region'] = True
	else:
		params['flag_check_constant_region'] = False

	if args.a:
		params['flag_check_3_adapter'] = True
	else:
		params['flag_check_3_adapter'] = False

	if args.sp:
		params['flag_check_spike_in'] = True
	else:
		params['flag_check_spike_in'] = False

	if args.p:
		params['pa_start'] = args.p

	f_log = make_log_file(out_log, p_params = params, p_vars = vars(args))
	
	'''
	pwrite(f_log, 'Input FASTQ file: ' + f_fastq)
	pwrite(f_log, 'Input intensity file: ' + f_intensity)
	pwrite(f_log, 'Position for the random region: ' + str(params['n_start']) + '~' + str(params['n_end']))
	if args.c:
		params['flag_check_constant_region'] = True
		pwrite(f_log, 'Check the constant region: Yes')
		pwrite(f_log, '\tconstant region sequence: ' + params['c_seq'])
		pwrite(f_log, '\tcorresponding position for the constant region sequence: ' + str(params['c_seq_pos']) + '~' + str(len(params['c_seq']) + params['c_seq_pos'] - 1))
		pwrite(f_log, '\tmaximal number of allowed mismatches: ' + str(params['c_hmdis_cutoff']))
		pwrite(f_log, '\tmaximal wiggle room: ' + str(params['c_wig']))
	else:
		params['flag_check_constant_region'] = False
		pwrite(f_log, 'Check the constant region: No')

	if args.a:
		params['flag_check_3_adapter'] = True
		pwrite(f_log, 'Check the 3\' adapter sequence: Yes')
		pwrite(f_log, '\tadapter sequence: ' + params['a_seq'])
		pwrite(f_log, '\tmaximal number of allowed mismatches: ' + str(int(len(params['a_seq']) * params['error_rate'])))
	else:
		params['flag_check_3_adapter'] = False
		pwrite(f_log, 'Check the 3\' adapter sequence: No')

	if args.sp:
		params['flag_check_spike_in'] = True
		pwrite(f_log, 'Check spike-in sequence: Yes')
		pwrite(f_log, '\tspike-in sequence: ' + params['sp_seq'])
		pwrite(f_log, '\tcorresponding position for the spike_in: ' + str(params['sp_start']) + '~' + str(len(params['sp_seq']) + params['sp_start'] - 1))
		pwrite(f_log, '\tmaximal number of allowed mismatches: ' + str(params['sp_hmdis_cutoff']))
		pwrite(f_log, '\tmaximal wiggle room: ' + str(params['sp_wig']))
	else:
		params['flag_check_spike_in'] = False
		pwrite(f_log, 'Check spike-in sequence: No')

	if args.p:
		params['pa_start'] = args.p
	pwrite(f_log, 'Starting position for the poly(A) region: ' + str(params['pa_start']))
	'''
	###-------------------------------------------------	
	# Filter reads from the fastq file based on the random region
	# master dictionary: {unique sequencer-id for each read : [sequence from the random region, lowest qc score for this region]}

	pwrite(f_log, '\nMaking a master dictionary with the fastq file...' + timer())
	
	# master dictionary, {idx, [seq, qc_min, hm_min, pos_offset]}
	mdict = {} 

	# dictionary for those having 3' adapter before the end of the random region (they have no poly(A) region)
	pdict ={}

	line_lst = []
	loss_ary = np.zeros(4)
	sp_count_ary = 0
	fq = fread(f_fastq)
	counting = 0
	rounds = 0
	counting_sum = 0
	counting_out = 0
	chunk_temp = params['chunk']
	while(True):
		line1 = fq.readline().rstrip()
		if not line1:
			with concurrent.futures.ProcessPoolExecutor(params['n_threads']) as pool:
				futures = pool.map(worker_fq,[line_lst[n:n+params['chunk_lines']] for n in range(0,len(line_lst),params['chunk_lines'])])
				for d1, d2, ary, sp_c in futures: # combine data from outputs from all processes
					mdict.update(d1)
					pdict.update(d2)
					loss_ary = loss_ary + ary
					sp_count_ary = sp_count_ary + sp_c
				counting_sum += counting
				pwrite(f_log, str(counting_sum) + ' reads processed...' + timer())
			break
		else:
			counting += 1
			line2 = fq.readline().rstrip()
			line3 = fq.readline().rstrip()
			line4 = fq.readline().rstrip()
			line_lst.append([line1, line2, line4])
			if counting % (params['chunk_lines'] * params['n_threads']) == 0:
				rounds += 1
				with concurrent.futures.ProcessPoolExecutor(params['n_threads']) as pool:
					futures = pool.map(worker_fq,[line_lst[n:n+params['chunk_lines']] for n in range(0,len(line_lst),params['chunk_lines'])])
					for d1, d2, ary, sp_c in futures: # combine data from outputs from all processes
						mdict.update(d1)
						pdict.update(d2)
						loss_ary = loss_ary + ary
						sp_count_ary = sp_count_ary + sp_c
					counting_sum = counting * rounds
					if counting_sum >= chunk_temp:
						chunk_temp += params['chunk']
						pwrite(f_log, str(counting_sum) + ' reads processed...' + timer())
				line_lst = []
				counting = 0
	pwrite(f_log, 'Total number of reads in the fastq file: ' + str(counting_sum))
	pwrite(f_log, 'Number of reads for checking intensities: ' + str(len(mdict)))
	if params['flag_check_spike_in']:
		pwrite(f_log, "Number of reads mapped to the spike-in: " + str(int(np.sum(sp_count_ary))))
		pwrite(f_log, "\t statistics of counts (hamming distance) for the spike-in: " + ', '.join([str(sp_count_ary[i])+'('+str(i)+')' for i in range(len(sp_count_ary))]))
	pwrite(f_log, "Number of tail-less reads (3' adapter before poly(A) start position): " + str(len(pdict)))
	pwrite(f_log, "Number of reads lost due to no match to the constant sequence region: " + str(int(loss_ary[0])))
	pwrite(f_log, "Number of reads lost due to too short random sequence region (<" + str(params['min_r_seq_len']) +") after trimming the 3' adapter: " + str(int(loss_ary[1])))
	pwrite(f_log, "Number of reads lost due to at least one N base present in the random sequence region: " + str(int(loss_ary[2])))
	pwrite(f_log, "Number of reads lost due to lowest QC in the random sequence region lower than " + str(params['qc_cutoff']) + ' :'  + str(int(loss_ary[3])))
	
	fq.close()

	# the number of variables in the value of master dictionary mdict
	# this is important for determing the start position of the A signal in the file
	for key in mdict:
		len_mdict_value = len(mdict[key])
		break

	# write out the reads that have recognized 3' adapters before poly(A) start position and manually assign tail length 0
	if len(pdict) > 0:
		with open(out_tl_less_tags, 'w') as f:
			f.write('id\tseq\tqc_min\thm_constant_region\tpos_offset\thm_3_adapter\tpos_idx_3_adapter\ttail_length\n')
			for key in pdict:
				f.write(key + '\t' + '\t'.join(map(str, pdict[key])) + '\t' + '0' + '\n')

		# also aggregate by random region and write out statistics which can be combined with the reads with poly(A) tails
		dict_pa_less = {}
		for key in pdict:
			if pdict[key][0] in dict_pa_less:
				dict_pa_less[pdict[key][0]].append(0)
			else:
				dict_pa_less[pdict[key][0]] = [0]

		with open(out_tl_less_stat, 'w') as f:
			f.write('Sequence' + '\t' + 'Median_TL' + '\t' + 'Duplicates' + '\t' + 'Q10' + '\t' + 'Q25' +'\t' + 'Q75' +'\t' + 'Q90' +'\n')
			for key, value in sorted(dict_pa_less.items(), key = lambda x: len(x[1]), reverse = True):
				f.write(key + '\t' + str(np.median(value)) + '\t' + str(len(value)) + '\t' \
				+ str(np.quantile(value, 0.1)) + '\t' + str(np.quantile(value, 0.25)) + '\t' \
				+ str(np.quantile(value, 0.75)) + '\t'+ str(np.quantile(value, 0.9)) + '\n')		



	# randomly pick a set for HMM training
	pwrite(f_log, 'Randomly picking training set:')
	if len(mdict) / 100 > params['training_max']:
		train_keys = random.sample(list(mdict.keys()), params['training_max'])
		pwrite(f_log, 'Dataset too large, ' + str(params['training_max']) + ' reads picked for training...')
	elif len(mdict) / 100 < params['training_min']:
		train_keys = random.sample(list(mdict.keys()), params['training_min'])
		pwrite(f_log, 'Dataset too small, ' + str(params['training_min']) + ' reads picked for training...')
	else:
		train_keys = random.sample(list(mdict.keys()), len(mdict) / 100)
		pwrite(f_log, str(len(mdict) / 100) + ' reads picked for training...')

	###-------------------------------------------------


	###------------------------------------------------
	# read intensity file and convert 4-channel intensities to single log-transformed bound A signal
	# output A signals to a file

	pwrite(f_log, '\nReading the intensity file...' + timer())	
	counting = 0
	rounds = 0
	counting_sum = 0
	counting_out = 0
	chunk_temp = params['chunk']
	train_set = []
	line_lst = []
	fi = fread(f_intensity)
	outA = open(out_A_signal, 'w')
	#outA.write('id' + '\t' + 'random_seq' + '\t' + 'qc_lowest' + '\t' + 'A_signal' + '\n')
	while(True):
		line = fi.readline()
		if not line:
			with concurrent.futures.ProcessPoolExecutor(params['n_threads']) as pool:
				futures = pool.map(worker_C2A,[line_lst[n:n+params['chunk_lines']] for n in range(0,len(line_lst),params['chunk_lines'])])
				for (d,l) in futures: # combine data from outputs from all processes
					train_set.extend(l)
					for key in d: # write converted A-signal to a file
						outA.write(key+'\t'+'\t'.join(map(str, d[key][:len_mdict_value]))+'\t'+'\t'.join(map(str,d[key][len_mdict_value]))+'\n')
						counting_out += 1
				counting_sum += counting
				pwrite(f_log, str(counting_sum) + ' reads processed...' + timer())
			break
		else:
			line_lst.append(line)
			counting += 1
			if counting % (params['chunk_lines'] * params['n_threads']) == 0:
				rounds += 1
				with concurrent.futures.ProcessPoolExecutor(params['n_threads']) as pool:
					futures = pool.map(worker_C2A,[line_lst[n:n+params['chunk_lines']] for n in range(0,len(line_lst),params['chunk_lines'])])
					for (d,l) in futures: # combine data from outputs from all processes
						train_set.extend(l)
						for key in d: # write converted A-signal to a file
							outA.write(key+'\t'+'\t'.join(map(str, d[key][:len_mdict_value]))+'\t'+'\t'.join(map(str,d[key][len_mdict_value]))+'\n')
							counting_out += 1
					counting_sum = counting * rounds
					if counting_sum >= chunk_temp:
						chunk_temp += params['chunk']
						pwrite(f_log, str(counting_sum) + ' reads processed...' + timer())
				line_lst = []
				counting = 0
	pwrite(f_log, 'The number of reads in training set after intensity-conversion: ' +  str(len(train_set)))
	pwrite(f_log, 'The total number of reads after intensity-conversion: ' + str(counting_out))
	pwrite(f_log, 'Finished processing the intensity file...' + timer())
	fi.close()
	outA.close()
	mdict.clear() # clear the dictionary to free up some memory
	###------------------------------------------------

###------------------------------------------------

else:
	out_A_signal = args.s
	if not os.path.isfile(out_A_signal):
		sys.exit('Error! No Tsignal file found!')
	# prefix for the output files
	if args.o:
		out_prefix = args.o
	else:
		out_prefix = out_A_signal.split('/')[-1].split('A_signal')[0]
	out_states = out_prefix + 'states.txt' # output file for HMM states
	out_hmm = out_prefix + 'HMM_model.txt' # output file for HMM model
	out_single_tags = out_prefix + 'pA_all_tags.txt' # output file for all tags with poly(A) tails (by HMM)
	out_pa_stat = out_prefix + 'pA_stat.txt' # output file for poly(A) tail length stats for each unique random sequence
	out_log = out_prefix + 'log.txt'

	f_log = open(out_log, 'w')
	pwrite(f_log, 'Starting HMM only mode...' + timer())
	# estimate the number of lines by dividing the total file size by the size of first line
	input_A_signal = fread(out_A_signal)
	line = input_A_signal.readline()
	len_mdict_value = line.rstrip().split('\t').index(str(params['bound'] * 100)) - 1
	line_size = int(input_A_signal.tell())
	input_A_signal.seek(0,2)
	file_size = int(input_A_signal.tell())
	T_lines = file_size / line_size
	pwrite(f_log, 'Estimated total number of reads in the Tsignal file: ' + str(T_lines))
	n_train_lines = min(max(T_lines/100, params['training_min']), params['training_max'])
	pwrite(f_log, str(n_train_lines) + ' reads picked for training...' + timer())
	input_A_signal.close()
	train_set = lines_sampler(out_A_signal, n_train_lines)


###------------------------------------------------
# initializes a gaussian hidden markov model and defines
# the tranisition, emission, and starting probabilities
pwrite(f_log, '\nTraining data with hmm...' + timer())
F = ghmm.Float()

pi = [1.0, 0.0, 0.0, 0.0, 0.0] # initial state

if params['allow_back'] == True:
	# The following matrix allows T states going back to non=T states.
	Transitionmatrix = [[0.04, 0.93, 0.02, 0.01, 0.0],
						[0.0, 0.87, 0.1, 0.02, 0.01],
         	           [0.0, 0.05, 0.6, 0.3, 0.05],
         	           [0.0, 0.01, 0.3, 0.6, 0.09],
         	           [0.0, 0.01, 0.01, 0.1, 0.88]]
else:
	# The following matrix does not allow states going backwards.
	Transitionmatrix = [[0.04, 0.93, 0.02, 0.01, 0.0],
						[0.0, 0.94, 0.03, 0.02, 0.01],
	                    [0.0, 0.0, 0.5, 0.4, 0.1],
	                    [0.0, 0.0, 0.0, 0.6, 0.4],
	                    [0.0, 0.0, 0.0, 0.0, 1.0]]
# state 0: peudo-A state
# state 1: definitive-A state
# state 2: likely-A state
# state 3: likely-non-A state
# state 4: definitive-non-A state

if params['mixed_model'] == True:
	Emissionmatrix = [[[params['bound']*100.0, 0.0], [1.0, 1.0], [1.0, 0.0]],
				  [[1.5, -1.0 ], [1.5, 1.5], [0.95, 0.05]],
                  [[1.5, -1.0 ], [1.5, 1.5], [0.75, 0.25]],
                  [[1.5, -1.0 ], [1.5, 1.5], [0.5, 0.5]],
                  [[1.5, -1.0 ], [1.5, 1.5], [0.25, 0.75]]]
	# [p1_mean, p2,mean], [p1_std, p2_std], [P(p1), P(p2)]
	model = ghmm.HMMFromMatrices(F, ghmm.GaussianMixtureDistribution(F), Transitionmatrix, Emissionmatrix, pi)
else:
	Emissionmatrix = [[params['bound']*100.0, 1.0],
					  [2.0, 0.5],
	                  [1.0, 0.5],
	                  [-1.0, 0.5],
	                  [-2.0, 0.5]]
	# [mean, std]
	model = ghmm.HMMFromMatrices(F, ghmm.GaussianDistribution(F), Transitionmatrix, Emissionmatrix, pi)

print('Model before training:')
print(model)
mghmm_train = ghmm.SequenceSet(F, train_set)
model.baumWelch(mghmm_train, 10000, 0.01)
print('Model after training:')
print(model)
model.write(out_hmm)

###------------------------------------------------

###------------------------------------------------
# calculate tail length using the mghmm model and write them to output files
# dict_tl structure: {gene_name : [list of tail lengths]}
pwrite(f_log, '\nCalculating tail-lengths and writing outputs...' + timer())
if params['flag_pa_override']:
	pwrite(f_log, 'Tail length override (by constant region and 3\' adapter position): ON')
else:
	pwrite(f_log, 'Tail length override (by constant region and 3\' adapter position): OFF')
lst_tl = [] # for storing random sequence, tail-length pairs
counting = 0
rounds = 0
counting_sum = 0
counting_out = 0
chunk_temp = params['chunk']
inA = fread(out_A_signal)
out_states = open(out_states, 'w')
out_states.write('id\tseq\thmm_states\n')
out_single_tags = open(out_single_tags, 'w')
out_single_tags.write('id\tseq\tqc_min\thm_constant_region\tpos_offset\thm_3_adapter\tpos_idx_3_adapter\ttail_length\n')
line_lst = []

while(1):
	line = inA.readline()
	if not line:
		with concurrent.futures.ProcessPoolExecutor(params['n_threads']) as pool:
			futures = pool.map(worker_hmm,[line_lst[n:n+params['chunk_lines']] for n in range(0,len(line_lst),params['chunk_lines'])])
			for (d, l) in futures: # combine data from outputs from all processes
				lst_tl.extend(l)
				for key in d: # write single tail tags to the output file
					out_states.write(key + '\t' + d[key][0] + '\t' + d[key][-1] + '\n')
					out_single_tags.write(key + '\t' + '\t'.join(d[key][:-1]) + '\n')
					counting_out += 1
			counting_sum += counting
			pwrite(f_log, str(counting_sum) + ' reads processed...' + timer())
			break
	else:
		line_lst.append(line)
		counting += 1
		if counting % (params['chunk_lines'] * params['n_threads']) == 0:
			rounds += 1
			with concurrent.futures.ProcessPoolExecutor(params['n_threads']) as pool:
				futures = pool.map(worker_hmm,[line_lst[n:n+params['chunk_lines']] for n in range(0,len(line_lst),params['chunk_lines'])])
				for (d, l) in futures: # combine data from outputs from all processes
					lst_tl.extend(l)
					for key in d: # write single tail tags to the output file
						out_states.write(key + '\t' + d[key][0] + '\t' + d[key][-1] + '\n')
						out_single_tags.write(key + '\t' + '\t'.join(d[key][:-1]) + '\n')
						counting_out += 1
				counting_sum = counting * rounds
				if counting_sum >= chunk_temp:
					chunk_temp += params['chunk']
					pwrite(f_log, str(counting_sum) + ' reads processed...' + timer()) 
			line_lst = []
			counting = 0
pwrite(f_log, 'Total number of tail-lengths written: ' + str(counting_out))
pwrite(f_log, 'Finished calculating tail lengths...' + timer())
inA.close()
out_states.close()
out_single_tags.close()
# out_stats file has the following: id, HMM states
# out_single_tags file has the following columns: id, random seq, lowest QC, hamming distance, postion offset, tail length

# delete temporary file containing converted T-signal (very big)
#subprocess.call(['rm','-f',Tsignal_file])

dict_tl = {} # dictionary for tail length, using random requence as key
for pair in lst_tl: # transform data for calculating median and mean tail length
	if pair[0] in dict_tl:
		dict_tl[pair[0]].append(pair[1])
	else:
		dict_tl.setdefault(pair[0],[pair[1]])

with open(out_pa_stat, 'w') as f:
	f.write('Sequence' + '\t' + 'Median_TL' + '\t' + 'Duplicates' + '\t' + 'Q10' + '\t' + 'Q25' +'\t' + 'Q75' +'\t' + 'Q90' +'\n')
	for key, value in sorted(dict_tl.items(), key = lambda x: len(x[1]), reverse = True):
		f.write(key + '\t' + str(np.median(value)) + '\t' + str(len(value)) + '\t' \
			+ str(np.quantile(value, 0.1)) + '\t' + str(np.quantile(value, 0.25)) + '\t' \
			+ str(np.quantile(value, 0.75)) + '\t'+ str(np.quantile(value, 0.9)) + '\n')		

pwrite(f_log, '\nTotal number of unique sequences with tail-length written: ' + str(len(dict_tl)))

pwrite(f_log, 'Final: ' + timer())		














		
		
		
		
		