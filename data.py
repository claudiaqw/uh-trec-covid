#!/usr/bin/env python
# coding: utf-8

import re
import os
import json
import gzip
import tarfile
import itertools as itt

from xml.etree import ElementTree
from tqdm import tqdm

METADATA = "./data/round2/metadata.csv"
TOPICS = "./data/round2/topics-rnd2.xml"
VALID_DOCS = "./data/round2/docids-rnd2.txt"
QRELS = "./data/round1/qrels-rnd2.txt"

def read_qrels(filepath = QRELS):
    result = {}
    res = []
    with open(filepath) as f:
        for line in tqdm(f, desc='loading qrels (by line)', leave=False):
            qid, _, docid, score = line.split()
            result.setdefault(qid, {})[docid] = score
            res.append([qid, docid, score])
    return res

def read_valid_docs(filepath = VALID_DOCS):
    with open(filepath) as f:
        doc_ids = [line.strip() for line in tqdm(f, desc='loading qrels (by line)', leave=False)]
    return doc_ids

def read_qrels_dict(file):
    result = {}
    file = open(file)
    for line in tqdm(file, desc='loading qrels (by line)', leave=False):
        qid, _, docid, score = line.split()
        result.setdefault(qid, {})[docid] = score
    return result

class Topic():
    def __init__(self, index, query, question, narrative):
        self.index = index
        self.query = query
        self.question = question
        self.narrative = narrative
        self.text = self.query + " " + self.question + " " + self.narrative

    def __str__(self):
        return self.index + " " + self.query

class TopicCollection():    
    def __init__(self, filepath = TOPICS):
        self.topics = {}
        tree = ElementTree.parse(filepath)
        for child in tree.getroot():
            n = child.attrib['number']
            query = child.find('query').text
            question = child.find('question').text
            narrative = child.find('narrative').text
            self.topics[n] = Topic(n, query, question, narrative)
     
    def __length__(self):
        return len(self.topics)

    def get_topic(self, idx):
        return self.topics[idx]



