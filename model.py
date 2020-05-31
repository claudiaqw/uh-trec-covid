#!/usr/bin/env python
# coding: utf-8

import torch
from transformers import BertTokenizer, BertModel

from utils import split_doc


class RankerManager():
    def __init__(self, ranking_model, queries, docs, valid_docs, qrel = None, output = "sim_output.txt", run_tag = "sim_run"):
        self.ranker = ranking_model
        self.topics = queries 
        self.docs = docs
        self.valid_docs = valid_docs        
        self.qrel = qrel
        self.output = output
        self.run_tag = run_tag
        
    def manage_rank(self):
        match_result = self.pair_doc_query()
        ranked_result = self.get_top_k(match_result, 5)
        self.export_result(ranked_result, self.output)
    
    def pair_doc_query(self):
        result = {}
        for qid in self.topics.topics:
            query = self.topics.get_topic(qid).text
            for i,docid in enumerate(self.valid_docs):
                print("Analyzing query %s, doc %s, %s " % (qid, i, docid))
                try:
                    doc = self.docs.get_document_from_jsom_no_paragraph_list(docid)['text']
                    score = self.rank(query, doc)
                    result.setdefault(qid, {})[docid] = score
                except Exception as e:
                    print(e)
        return result    
    
    def rank(self,query, doc):
        return self.ranker.rank(query, doc)
       
    def get_top_k(self, result, k = 500):
        final_result = []
        for qid, scored_docs in result.items():
            sorted_docs = {d:s for d,s in sorted(scored_docs.items(), key=lambda item: item[1], reverse=True)}
            firstkpairs = {d: sorted_docs[d] for d in list(sorted_docs)[:k]}
            tmp = [(qid, docid, score, i) for i, (docid, score) in enumerate(firstkpairs.items(), start=1)]
            final_result += tmp
        return final_result
    
    #topicid Q0 docid rank score run-tag
    def export_result(self, result, output_file): 
        with open(output_file, mode='w+b') as f:
            for (qid, docid, score, pos) in result:
                line = bytes('%s Q0 %s %s %s %s\n' % (qid, docid, pos, score, self.run_tag), 'utf-8')
                f.write(line )    

class BertSimilarity():
    def __init__(self, pretrained_model = 'bert-base-uncased'):
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model)
        self.model = BertModel.from_pretrained(pretrained_model)
   
    def _split_doc(self, query, doc):    
        tok_query = self.tokenizer.tokenize(query)
        tok_doc = self.tokenizer.tokenize(doc)
        
        query_len = len(tok_query)
        adit_len = 3
        maxlen = self.model.config.max_position_embeddings
        max_doc_len = maxlen - query_len - adit_len
        
        doc_chunks, _ = split_doc(tok_doc, max_doc_len)
        return tok_query, doc_chunks        
    
    def _format_input(self, query_tok, doc_tok):
        tokenized_text = ["[CLS] "] + query_tok + [" [SEP]"] + doc_tok + [" [SEP]"]

        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        segments_ids = [1] * (len(query_tok) + 2) + [0] * (len(doc_tok) + 1)

        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])
        sep_idx = len(query_tok)
        return tokens_tensor, segments_tensors, sep_idx        
      
    def _embed(self, query_tok, doc_tok):
        tokens_tensor, segments_tensors, sep_index = self._format_input(query_tok, doc_tok)
        
        self.model.eval()
        with torch.no_grad():
            last_hidden_state,_,_ = self.model(tokens_tensor, segments_tensors)
            
        last_hidden_state = last_hidden_state.squeeze(dim = 0)
        query_embed = last_hidden_state[1: sep_index]
        doc_embed = last_hidden_state[sep_index + 1:]

        query_embed = torch.sum(query_embed, dim = 0)
        doc_embed = torch.sum(doc_embed, dim=0)        
        return query_embed, doc_embed
    
    def _calculate_similarity(self, query_embed, doc_embed):
        from scipy.spatial.distance import cosine
        return 1 - cosine(query_embed, doc_embed)
            
    def rank(self, query, doc):        
        query_tok, splitted_doc_tok = self._split_doc(query, doc)
             
        queries, docs = [], []
        for doc in splitted_doc_tok:
            q, d = self._embed(query_tok, doc)
            queries.append(q.unsqueeze(dim=0))
            docs.append(d.unsqueeze(dim=0))            
       
        query_embed = torch.cat(queries, dim= 0).mean(dim=0)
        doc_embed = torch.cat(docs, dim= 0).mean(dim=0)
        return self._calculate_similarity(query_embed, doc_embed)