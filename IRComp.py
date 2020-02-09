# -*- coding: utf-8 -*-
"""
IRcomp
"""

import pandas as pd
import numpy as np
import os,math,csv

sample_queries=xrange(301,351)
sample_queries1=xrange(601,701)

def run_query(res_file=None):
	if res_file==None: return 'res_file_1000'
	os.system(" ./../indri-5.8-install/bin/IndriRunQuery ./indriRunQuery_psf_2.xml ./queriesROBUST.xml > " + res_file)
	return res_file

def create_ret_file(new_relevant_docs,res_file='new_res_1000'):
	ret=pd.read_csv(new_relevant_docs, index_col=False, sep=' ', header=None)[:1000]
	ret.to_csv('new_res_1000', sep=' ', index=False, header=None)
	return 'new_res_1000'

	   
def ret_eval(res_file='res_q',eval_file='res_eval'):
	os.system("trec_eval_9.0/trec_eval qrels_50_Queries " + res_file + "| grep all | grep map > " + eval_file)
	eval_f=pd.read_csv(eval_file, sep=' ', header=None, names=['name','all','value'])

	return eval_f['value'][0]


#this calles cluster for a query
def cluster_docs(all_q_relevant_docs, term_dict, all_docs_vecs):

	#for query in set(all_q_relevant_docs['query']):
	for query in sample_queries:
		fifty_relevant_docs=all_q_relevant_docs[all_q_relevant_docs['query']==query][:50] 
		docs_order=[]
		
		for index,row in fifty_relevant_docs.iterrows():
			docs_order.append(row['docno'])

		new_order_fifty_relevant=cluster_for_query(query, docs_order, all_docs_vecs, term_dict)
		
		all_q_relevant_docs[all_q_relevant_docs['query']==query][:50]=new_order_fifty_relevant
		
		all_q_relevant_docs.to_csv('new_relevant_docs', sep=' ', header=None, index=False)
	return 'new_relevant_docs'
	
	
def cluster_for_query(query, docs_order, all_docs_vecs, term_dict):

	docs_for_query={}
	
	for docno in docs_order:
		if os.path.isfile('all_docs_vecs')==False:
			bla=find_doc_features(docno, term_dict, all_docs_vecs)
			all_docs_vecs[docno]=bla
			
		docs_for_query[docno]=all_docs_vecs[docno]
		
	new_order_fifty_relevant=KNN(docs_for_query, docs_order)
	 
	return new_order_fifty_relevant
  
	
def KNN(docs_for_query_tfidf, docs_order, k=6):
	k_nearest={}

	for docno in docs_order:
		k_nearest[docno]=find_k_nearest(docno,docs_for_query_tfidf,k)
	
	docs_new_order=[]
	for i in range(50):
		near_i =k_nearest[docs_order[i]]
		for j in range(5):
			if near_i[j] in docs_new_order: continue
			docs_new_order.append(near_i[j])

	return np.asarray(docs_new_order)
	
def cosine(vec_docno1, vec_docno2): 
	return (vec_docno1 * vec_docno2).sum(0, keepdims=True) ** .5     

	
def find_k_nearest(docno,docs_for_query,k):
	temp_sim=[]
	for doc1 in docs_for_query:
		temp_sim.append(doc1,cosine(docs_for_query[docno],docs_for_query[doc1]))
	temp_sim.sort(key=lambda tup: tup[1])[:k+1]
	
def find_doc_features(docno, term_dict, all_docs_vecs):
	
	# get doc number:
	os.system(" ./../indri-5.8-install/bin/dumpindex  ../../../data/ROBUST/ROBUSTindex di docno " + docno + " > temp.txt")
	docid=None
	with open('temp.txt', 'r') as f:
		docid=f.readlines()[0][:-1]

				
	f.close

	#print 'docno, docid', docno, docid, len(docid)
	all_docs_vecs[docno]=create_docs_tfidf(docno, docid, term_dict) # unorganized dict of all terms
   
	doc_vec_tfidf=[] # organized dict of all terms
	for term in term_dict:
		doc_vec_tfidf.append(all_docs_vecs[docno][term])
   
	return doc_vec_tfidf

# only here ther is need for docid
def create_docs_tfidf(docno, docid,term_dict): 
	docid_file_name= str(docid)
	print 'docid_file_name', docid_file_name, len(docid_file_name)
   #raw_input()
	doc_dict={}
	if os.path.isfile(docid_file_name)==False:
		os.system(" ./../indri-5.8-install/bin/dumpindex  ../../../data/ROBUST/ROBUSTindex dv " + str(docid) + " > " + docid_file_name)
	
	docid_dv=pd.read_csv(docid_file_name, sep=' ', header=None, names=['position','first_position','word'])
	#print 'set(docid_dv[word], docno', docid_dv, docno
	num_terms_in_document=len(set(docid_dv['word']))
	#print term_dict
	for term in term_dict:
		#print 'term', term
		occur=len(docid_dv[docid_dv['word']==term])
		num_docs_with_t=term_dict[term][0]
		#print 'term, occur', term, occur, num_terms_in_document
		doc_dict[term]=(float(occur)/num_terms_in_document,math.log((528155/num_docs_with_t)))
		
	#print doc_dict
	return doc_dict


def rearrange_docs(): 

	#term_dict=create_vocabulary()

	if os.path.isfile('all_docs_vecs')==False: all_docs_vecs={}
	else: all_docs_vecs=pd.to_dict('all_docs_vecs')
	
	res_file_1000=run_query() 
	map1=ret_eval(res_file_1000)
	
	all_q_relevant_docs=pd.read_csv(res_file_1000, sep=' ',index_col=False, header=None,names=['query', 'b1','docno','b2','b3','b4'])
	
	new_relevant_docs=cluster_docs(all_q_relevant_docs, term_dict, all_docs_vecs)
	pd.DataFrame(all_docs_vecs).to_csv('all_docs_vecs')
	
	map2=ret_eval(new_relevant_docs)
	
	#print (map1,map2)
	if map1<map2: create_ret_file(new_relevant_docs)


def create_vocabulary():
	term_dict={} 
	if os.path.isfile('vocabulary_stats')==False:
		os.system(" ./../indri-5.8-install/bin/dumpindex  ../../../data/ROBUST/ROBUSTindex v  > vocabulary_stats")
	terms_list=pd.read_csv('vocabulary_stats', sep=' ', index_col=[0], header=None, names=['word','tot_occur','num_docs_with_t'])
	f = open('mycsvfile1','wb')
	w = csv.writer( f, delimiter=' ' )
	#w.writeheader()

	for index, row in terms_list.iterrows():
		if index=='TOTAL': continue
		
		#if str(index).isdigit(): continue
		if int(row['num_docs_with_t'])==1 & int(row['tot_occur'])<=5: continue

		term_dict[index]=(int(row['num_docs_with_t']),int(row['tot_occur']))
		#print index, '\n' ,term_dict[index]
		w.writerow((index, row['num_docs_with_t'],row['tot_occur']))
	

	f.close()
	return term_dict


create_vocabulary()
#rearrange_docs()