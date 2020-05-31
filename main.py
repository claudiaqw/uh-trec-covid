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

cov_dm = TrecCovidDatasetManager(DOCS, METADATA)
cov_dm.load_metadata_from_csv()
cov_dm.save_metadata_as_pickle()

manager = RankerManager(ranking_model, queries, cov_dm, valid_docs)
manager.manage_rank()