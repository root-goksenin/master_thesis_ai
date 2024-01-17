"""
This example show how to evaluate BM25 model (Elasticsearch) in BEIR.
To be able to run Elasticsearch, you should have it installed locally (on your desktop) along with ``pip install beir``.
Depending on your OS, you would be able to find how to download Elasticsearch. I like this guide for Ubuntu 18.04 -
https://linuxize.com/post/how-to-install-elasticsearch-on-ubuntu-18-04/ 
For more details, please refer here - https://www.elastic.co/downloads/elasticsearch. 

This code doesn't require GPU to run.

If unable to get it running locally, you could try the Google Colab Demo, where we first install elastic search locally and retrieve using BM25
https://colab.research.google.com/drive/1HfutiEhHMJLXiWGT8pcipxT5L2TpYEdt?usp=sharing#scrollTo=nqotyXuIBPt6


Usage: python evaluate_bm25.py
"""


from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.lexical import BM25Search as BM25

import json
import logging

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

data_name = "fiqa"
data_path = f"/home/gyuksel/master_thesis_ai/gpl_given_data/{data_name}"
corpus, queries, qrels = GenericDataLoader(data_path).load(split="dev")

#### Lexical Retrieval using Bm25 (Elasticsearch) ####
#### Provide a hostname (localhost) to connect to ES instance
#### Define a new index name or use an already existing one.
#### We use default ES settings for retrieval
#### https://www.elastic.co/

hostname = "localhost" #localhost
index_name = data_name 

#### Intialize #### 
# (1) True - Delete existing index and re-index all documents from scratch 
# (2) False - Load existing index
initialize = False # False

#### Sharding ####
# (1) For datasets with small corpus (datasets ~ < 5k docs) => limit shards = 1 
# SciFact is a relatively small dataset! (limit shards to 1)
model = BM25(index_name=index_name, hostname=hostname, initialize=initialize, timeout=10000)

# (2) For datasets with big corpus ==> keep default configuration
# model = BM25(index_name=index_name, hostname=hostname, initialize=initialize)
retriever = EvaluateRetrieval(model, k_values=[9999])

#### Retrieve dense results (format of results is identical to qrels)
results = retriever.retrieve(corpus, queries)

with open(f"bm25_scores/{index_name}/dev_bm25_scores.json", "w") as f:
    json.dump(results, f)
