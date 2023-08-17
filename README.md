# MPRA_tail_seq
Massively parallel reporter assay with tail-length sequencing 

This script has two modes:
> Mode 1 takes two files: python -u script.py file1 file2
1. fastq file from read1 
2. intensity file from read1

> Mode 2 takes 1 file: python -u script.py file1
1. A_signal output file generated by this script
Note: When using LSF, make sure to add -n 20 for multiprocessing

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
