from data import TopicCollection, read_valid_docs
from model import RankerManager, BertSimilarity
from data_loader import TrecCovidDatasetManager
import random

TOPICS = './data/round3/topics-rnd3.xml'
VALID_DOCS = './data/round3/docids-rnd3.txt'
METADATA = './data/round3/metadata.csv'
DOCS = './data/round3/'

ranking_model = BertSimilarity('./pretrained_models/scibert_scivocab_uncased')
queries = TopicCollection(TOPICS)

# Reading the data set/////////////////////////////////////////////////////////////////////////////////////////
cov_dm = TrecCovidDatasetManager(DOCS, METADATA)
cov_dm.load_metadata_from_csv_round3()
# cov_dm.load_metadata_from_csv_round2()

# Saving metadata dictionary for future uses
cov_dm.save_metadata_as_pickle()
# cov_dm.load_metadata_from_pickle()

# Creating a documents dict to speed up the process
cov_dm.create_papers_dict()

# Saving the documents dict for future uses
cov_dm.save_docs_dict_as_pickle()
# cov_dm.load_docs_dict_from_pickle()

# Getting the valid docs/////////////////////////////////////////////////////////////////////////////////////
valid_docs = cov_dm.get_valid_docs()
valid_docs = random.sample(valid_docs, 1000)
# valid_docs = ['wp1hd5w9', '6qpsxmgi']

# Creating the Ranking///////////////////////////////////////////////////////////////////////////////////////

manager = RankerManager(ranking_model, queries, cov_dm, valid_docs)
manager.manage_rank()
