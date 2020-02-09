# this will be lm model retrieval and lm model clustering
#inspired by: https://github.com/liheyuan/SimpleLMIR

import pandas as pd
import numpy as np
import os, math, csv,json, io
from operator import itemgetter
try:
    to_unicode = unicode
except NameError:
    to_unicode = str

	

sample_queries=xrange(301,351)
sample_queries1=xrange(301,701)
QUERIES={}
#QUERIES={301:"international organized crime", 302:"poliomyelitis post polio",303: "hubble telescope achievements", 304:"endangered species mammals"}
with open('QUERIES.json') as data_file:
	QUERIES = json.load(data_file)

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
	ptq = 1/ float(len(vec_docno1))
	
	for word in vec_docno1:
		ptd=float(vec_docno2[word])/len(vec_docno2)
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
		
		if all_docs_vecs=={}:
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

#################### LM part ######################
def create_lm(all_q_relevant_docs, term_dict, all_docs_vecs):

	for docno in all_q_relevant_docs:
	
		bla=find_doc_features(docno, term_dict, all_docs_vecs)
		all_docs_vecs[docno]=bla
	return all_docs_vecs

def lm_ret(all_docs_vecs, all_q_relevant_docs):
	for query in sample_queries:
		print 'query', query
		rank=RankKL(query,all_docs_vecs)
		unzipped = zip(*rank )

		all_q_relevant_docs.loc[all_q_relevant_docs['query']==query, 'docno'][:1000]=np.asarray(unzipped[0])
		all_q_relevant_docs.loc[all_q_relevant_docs['query']==query, 'b3'][:1000]=np.asarray(unzipped[1])
		
		all_q_relevant_docs.to_csv('relevant_docs', sep=' ', header=None, index=False)
	return 'relevant_docs'

def get_text(query):
	return QUERIES[str(query)]
	
# Rank doc in colls according to query, score by Kullback-Leibler Divergence(KLD)
def RankKL( query,all_docs_vecs ):
	# Analysis query words
	query_text = get_text(query).split(' ')
	num_docs=len(all_docs_vecs)
	# Score each doc
	result = []
	for docno in all_docs_vecs:
		score = 0.0
		# score += -p(t|q)*log(P(t|d)) 
		ptq = float(1) / float(len(query_text))
		for word in query_text: 

			'''if all_docs_vecs[docno].has_key(word)==False:continue
			ptd = float(all_docs_vecs[docno][word])/len(all_docs_vecs[docno])
			if ptd == 0.0:  continue
			lptd = math.log(ptd, 2)'''
			
			psf=0
			alpha=0.5
			for doc1 in all_docs_vecs:
				if all_docs_vecs[doc1].has_key(word)==False:continue
				ptd1 = float(all_docs_vecs[doc1][word])/len(all_docs_vecs[doc1])
				pd1=float(1)/num_docs
				pi=1
				for word1 in query_text:
					if all_docs_vecs[doc1].has_key(word1)==False:continue					
					ptd_tag=float(all_docs_vecs[doc1][word1])/len(all_docs_vecs[doc1])
					pi*=ptd_tag
				
				psf+=ptd1*pd1*pi
			
			score += alpha*ptq+(1-alpha)*psf
		# Add to result
		result.append((docno, score))
	# Sort & return

	return sorted(result,key=itemgetter(1), reverse=True)[:1000]
		
		
def rearrange_docs(): 

	#term_dict=create_vocabulary()
	terms_list=pd.read_csv('term_list_digits', sep=' ', index_col=[0])
	all_docs_vecs={}
	if os.path.isfile('all_docs_vecs.json'):
		with open('all_docs_vecs.json') as data_file:
			all_docs_vecs = json.load(data_file)
	
	res_file='psf_2_res_0.2452_2000'
	map1=0.2421
	
	all_q_relevant_docs=pd.read_csv(res_file, sep=' ',index_col=False, header=None,names=['query', 'b1','docno','b2','b3','b4'])
	

	#lm part:
	'''docs=set(all_q_relevant_docs['docno'])
	docs.remove('in')
	docs.remove('language')
	all_docs_vecs=create_lm(docs, terms_list, all_docs_vecs)'''
	'''with io.open('all_docs_vecs.json', 'w', encoding='utf8') as outfile:
		str_ = json.dumps(all_docs_vecs,
                      indent=4, sort_keys=True,
                      separators=(',', ': '), ensure_ascii=False)
		outfile.write(to_unicode(str_))
	print 'after create_lm'''
	lm_ret_file=lm_ret(all_docs_vecs, all_q_relevant_docs)
	
	new_relevant_docs=cluster_docs(lm_ret_file, terms_list, all_docs_vecs,50,6)
	
	
	
	map2=ret_eval(new_relevant_docs)
	
	print (map1,map2)
 


rearrange_docs()