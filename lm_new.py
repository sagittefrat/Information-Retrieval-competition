# this will be lm model retrieval and lm model clustering
#inspired by: https://github.com/liheyuan/SimpleLMIR

import pandas as pd
import numpy as np
import os, math, csv
import comp

# -*- coding: utf-8 -*-
"""
IRcomp
"""
#remove numbers and words that exists only in 1 document so cluster will be faster


import pandas as pd
import numpy as np
import os, math, csv

sample_queries=xrange(301,351)
sample_queries1=xrange(301,701)
QUERIES=("international organized crime", "poliomyelitis post polio", "hubble telescope achievements", "endangered species mammals")  

def find_doc_features(docno, term_dict, all_docs_vecs):
	
	# get doc number:
	os.system(" ./../indri-5.8-install/bin/dumpindex  ../../../data/ROBUST/ROBUSTindex/ di docno " + docno + " > temp.txt")
	docid=None
	with open('temp.txt', 'r') as f:
		docid=f.readlines()[0][:-1]			
	f.close

	all_docs_vecs[docno]=docno_word_dict(docid, term_dict)
   
	return all_docs_vecs[docno]


def docno_word_dict(docid, term_dict):
	docid_file_name= str(docid)

	doc_dict={}
	
	os.system(" ./../indri-5.8-install/bin/dumpindex  ../../../data/ROBUST/ROBUSTindex/ dv " + str(docid) + " > " + docid_file_name)
	
	docid_dv=pd.read_csv(docid_file_name, sep=' ', header=None, names=['position','first_position','word'])
	num_terms_in_document= len(set(docid_dv['word']))
	for index, row in docid_dv.iterrows(): 
		if doc_dict.has_key(row['word']): continue
		if row['word']=='---': continue
		if row['word'] in term_dict.index:

			occur=len(docid_dv[docid_dv['word']==row['word']])
			num_docs_with_t=term_dict.loc[row['word']]['num_docs_with_t']
			doc_dict[row['word']]=(float(occur)/num_terms_in_document*math.log((528155/num_docs_with_t)))

	os.remove(docid_file_name)
	
	return doc_dict
	
# this is sim via lm
def sim(vec_docno1, vec_docno2, doc1, doc2):
	score = 0.0
	# score += -p(t|q)*log(P(t|d)) 

	for word in vec_docno1:
		ptq = 1/ float(len(vec_docno1))
		ptd=vec_docno2[word])/len(vec_docno2)
		
		if ptd == 0.0:continue
		
		lptd = math.log(ptd, 2)
		score += -ptq*lptd	
	
	return score

	
def find_k_nearest(docno,docs_for_query,k):

	temp_sim=[]
	for doc2 in docs_for_query:
		if docs_for_query[doc2]=={}: print doc2
		
		temp_sim.append((doc2,sim(docs_for_query[docno],docs_for_query[doc2], docno, doc2)))

	temp_sim.sort(key=lambda tup: tup[1], reverse=True)
	return temp_sim[:k]
	
def KNN(docs_for_query, docs_order, k=6):
	k_nearest={}

	for docno in docs_order:
		k_nearest[docno]=find_k_nearest(docno,docs_for_query,k)
	
	docs_new_order=[]
	for i in range(len(docs_for_query)):
		near_i =k_nearest[docs_order[i]]

		for j in range(k):
		
			#if near_i[j][1]<0.2: continue
			if near_i[j][0] in docs_new_order: continue
			
			docs_new_order.append(near_i[j][0])

	return np.asarray(docs_new_order)



def cluster_for_query(query, docs_order, all_docs_vecs, term_dict,k=None,num_clusters=None):
	
	docs_for_query={}
	new_all_relevant=docs_order
	for docno in docs_order:
		
		if os.path.isfile('all_docs_vecs')==False:
			bla=find_doc_features(docno, term_dict, all_docs_vecs)
			all_docs_vecs[docno]=bla
			
		docs_for_query[docno]=all_docs_vecs[docno]
	if k!=None:
		new_order_fifty_relevant=KNN(docs_for_query, docs_order,k)
	elif num_clusters!=None:
		new_order_fifty_relevant=LM(docs_for_query, docs_order,num_clusters)
	
	return new_order_fifty_relevant


#this calles cluster for a query
def cluster_docs(all_q_relevant_docs, term_dict, all_docs_vecs, num_docs_to_cluster=50,k=None,num_clusters=None):

	
	for query in sample_queries1:
		print 'query', query
		fifty_relevant_docs=all_q_relevant_docs[all_q_relevant_docs['query']==query][:num_docs_to_cluster]
	
		docs_order=[]
		
		for index,row in fifty_relevant_docs.iterrows():
			docs_order.append(row['docno'])
		np.asarray(docs_order)

		new_order_all_relevant=np.asarray(cluster_for_query(query, docs_order, all_docs_vecs, term_dict,k))

		all_relevant=all_q_relevant_docs[all_q_relevant_docs['query']==query]['docno'].as_matrix()
		all_relevant[:num_docs_to_cluster]=new_order_all_relevant

		all_q_relevant_docs.loc[all_q_relevant_docs['query']==query, 'docno']=all_relevant
		
		all_q_relevant_docs.to_csv('new_relevant_docs', sep=' ', header=None, index=False)
	
	return 'new_relevant_docs'

def ret_eval(res_file='res_file_1000',eval_file='res_eval'):
	os.system("../trec_eval_9.0/trec_eval qrels_50_Queries " + res_file + "| grep all | grep map > " + eval_file)
	eval_f=pd.read_csv(eval_file, sep='\t', header=None, names=['name','all','value'])

	return eval_f['value'][0]

def rearrange_docs(): 

	#term_dict=create_vocabulary()
	terms_list=pd.read_csv('term_list_digits', sep=' ', index_col=[0])
	all_docs_vecs={}

	
	res_file='psf_2_res_0.2452_2000'
	map1=0.2421
	
	all_q_relevant_docs=pd.read_csv(res_file, sep=' ',index_col=False, header=None,names=['query', 'b1','docno','b2','b3','b4'])
	
	new_relevant_docs=cluster_docs(all_q_relevant_docs, terms_list, all_docs_vecs,50,6)

	
	map2=ret_eval(new_relevant_docs)
	
	print (map1,map2)
 


rearrange_docs()