        
        
        
from easy_elasticsearch import ElasticSearchBM25
import numpy as np 
from gpl_improved.hard_negative_miner import HardNegativeMiner
from gpl_improved.trainer.RetriverWriter import EvaluateGPL
from gpl_improved.utils import load_sbert
from beir.datasets.data_loader import GenericDataLoader
import time
import tqdm
from functools import partial
import json 
from concurrent.futures import ThreadPoolExecutor



def get_doc(corpus, doc_id):
    return " ".join([corpus[doc_id]["title"], corpus[doc_id]["text"]])

if __name__ == "__main__":
        
        corpus, queries, qrels = GenericDataLoader("./gpl_given_data/scifact").load(split="test")
        bi_retriver = load_sbert("GPL/msmarco-distilbert-margin-mse", pooling = None, max_seq_length = 350).cuda()
        retriver = EvaluateGPL(bi_retriver,queries, corpus)
        result = retriver.results()
        partial_get_doc = partial(get_doc, corpus)
        docs = list(map(partial_get_doc, corpus.keys()))
        dids = np.array(list(corpus.keys()))
        pool = dict(zip(dids, docs))
        bm25 = ElasticSearchBM25(
            pool,
            port_http="9222",
            port_tcp="9333",
            service_type="executable",
            index_name=f"one_trial{int(time.time() * 1000000)}",
        )
        def calculate_score(q_id):
            return bm25.score(q_id, corpus.keys())

        
        
        scores = {q_id: calculate_score(q_id) for q_id in tqdm.tqdm(result.keys())}
        for q_id in tqdm.tqdm(result.keys()): 
            # Get the document scores for the given query from dense retriver. This contains all the document ids
            dense_score = result[q_id]
            bm25_score = scores[q_id]
            scores[q_id] = {key: dense_score[key] * bm25_score[key] for key in dense_score}
            # Score the same query against every document in the corpus using bm25
            # Update the document scores from dense retriver by multiplying with bm25 scores. 
            # If BM25 score is 0, then keep the dense retriver score.
        with open("results.json", "w") as f:
            json.dump(result, f)
        with open("new_results.json", "w") as f:
            json.dump(scores, f)
        # for qid, pos_dids in tqdm.tqdm(gen_qrels.items()):
        #     query = gen_queries[qid]
        #     rank = bm25.query(query, topk=50)  # topk should be <= 10000
        #     print(rank.keys())
        #     neg_dids = list(rank.keys())
        #     for pos_did in gen_qrels[qid]:
        #         if pos_did in neg_dids:
        #             neg_dids.remove(pos_did)
        #     result[qid] = neg_dids
