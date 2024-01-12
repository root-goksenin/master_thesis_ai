
from gpl_improved.trainer.RetriverWriter import RetriverWriter, EvaluateGPL
from gpl_improved.utils import load_sbert
from beir.datasets.data_loader import GenericDataLoader
import json 


with open("new_results.json") as f:
    results = json.load(f)


corpus, queries, qrels = GenericDataLoader("./gpl_given_data/scifact").load(split="test")
bi_retriver = load_sbert("GPL/msmarco-distilbert-margin-mse", pooling = None, max_seq_length = 350)
retriver = EvaluateGPL(bi_retriver,queries, corpus)
retriver.results_ = results 
evals = RetriverWriter(retriver, "configs")
evals.evaluate_beir_format(qrels)