from data import TopicCollection, read_valid_docs
from model import RankerManager, BertSimilarity
from trec_covid_data_loader import TrecCovidDatasetManager

TOPICS = './data/round2/topics-rnd2.xml'
VALID_DOCS = './data/round2/docids-rnd2.txt'
METADATA = './data/round2/metadata.csv'
DOCS = './data/round2/'

ranking_model = BertSimilarity('./pretrained_models/scibert_scivocab_uncased')
queries = TopicCollection()
#valid_docs = read_valid_docs()
valid_docs = ['wp1hd5w9', '6qpsxmgi']


# Reading the dataset
cov_dm = TrecCovidDatasetManager(DOCS, METADATA)
cov_dm.load_metadata_from_csv()

# Saving metadata dictionary for future uses
cov_dm.save_metadata_as_pickle()
# cov_dm.load_metadata_from_pickle()

# Creating a documents dict to speed up the process
cov_dm.create_papers_dict()

# Saving the documents dict for future uses
cov_dm.save_docs_dict_as_pickle()
# cov_dm.load_docs_dict_from_pickle()



manager = RankerManager(ranking_model, queries, cov_dm, valid_docs)
manager.manage_rank()