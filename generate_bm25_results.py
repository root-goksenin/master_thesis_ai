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


from beir import LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.lexical import BM25Search as BM25
from gpl_improved.trainer.RetriverWriter import RetriverWriter, EvaluateGPL
import json
import logging
import click
import os
#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout
logger = logging.getLogger(__name__)



@click.command()
@click.option("--data_name", type = str)
def main(data_name):
    data_path = f"/home/gyuksel/master_thesis_ai/{data_name}"
    corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")
    hostname = "localhost"
    index_name = os.path.split(data_name)[1]
    logger.info(f"BM25 for {data_path} as index {index_name}")
    initialize = False
    model = BM25(index_name=index_name, hostname=hostname, initialize=initialize, timeout=10000)
    # Normally EvaluateGPL is used for evaluating dense models. So we need this hacky way of getting the BM25 model into EvaluateGPL
    evaluator = EvaluateGPL(model = model, query=queries, corpus= corpus)
    # A bit hacky way of getting BM25 into the EvaluateGPL
    evaluator.retriever = EvaluateRetrieval(model, k_values=[1000])
    # Init Writer, and write scores.
    writer = RetriverWriter(evaluator, output_dir=f"bm25_scores/{index_name}", write_scores=True)
    writer.evaluate_beir_format(qrels)
    
if __name__ == "__main__":
    main()