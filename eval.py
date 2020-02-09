# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np

sample_queries=xrange(301,351)

eval_array=np.zeros(shape=(50))

res_file_old=pd.read_csv('res_file_1000', sep=' ', names=['query', 'b1','docno','b2','b3','b4'])
res_file_new=pd.read_csv('new_relevant_docs', sep=' ',names=['query', 'b1','docno','b2','b3','b4'])
 
def eval(file1, file2) : 
	for query in sample_queries:
		'''file1[file1['query']==query].to_csv('temp_res', sep=' ', header=None, index=False)
		os.system('../trec_eval_9.0/trec_eval qrels_50_Queries temp_res | grep all | grep map >  eval_file1')
		eval_f1=pd.read_csv('eval_file1', sep= '\t', header=None, names=['name','all','value'])'''

		file2[file2['query']==query].to_csv('temp_res', sep=' ', header=None, index=False)
		os.system('../trec_eval_9.0/trec_eval qrels_50_Queries temp_res | grep all | grep map >  eval_file2')
		eval_f2=pd.read_csv('eval_file2', sep= '\t', header=None, names=['name','all','value'])

		#eval_array[query-301] = ( int(query), eval_f1['value'][0], eval_f2['value'][0] ) 
		eval_array[query-301] = ( eval_f2['value'][0] ) 
		
		bla=pd.DataFrame(eval_array)
		bla.to_csv('temp_res', sep='\t', header=['new'], index=False)

eval(res_file_old, res_file_new)