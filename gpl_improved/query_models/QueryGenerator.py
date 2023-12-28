from enum import Enum
from enum import auto
from tqdm.autonotebook import trange
from beir.util import write_to_json, write_to_tsv
from typing import Dict
import logging, os
from gpl_improved.query_models import QAugmentModel


class QueryGenerator:
    def __init__(self, model, augment_model: QAugmentModel, **kwargs):
        self.model = model
        self.augment_model = augment_model
        self.qrels = {}
        self.queries = {}
        self.augmented_queries = {}
        self.aug_to_q = {}
        self.logger = logging.getLogger(__name__ + ".QueryGenerator")


    @staticmethod
    def save(output_dir: str, queries: Dict[str, str], qrels: Dict[str, Dict[str, int]], prefix: str):
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, prefix + "-qrels"), exist_ok=True)
        
        query_file = os.path.join(output_dir, prefix + "-queries.jsonl")
        qrels_file = os.path.join(output_dir, prefix + "-qrels", "train.tsv")
        
        write_to_json(output_file=query_file, data=queries)
        
        write_to_tsv(output_file=qrels_file, data=qrels)

    @staticmethod
    def save_query_aug(output_dir: str, query_aug: Dict[str, str], prefix: str):
        
        os.makedirs(output_dir, exist_ok=True)
  
        query_aug_file = os.path.join(output_dir, prefix + "-aug.jsonl")

        write_to_json(output_file=query_aug_file, data=query_aug)
        
    def generate(self, 
                 corpus: Dict[str, Dict[str, str]], 
                 output_dir: str, 
                 top_p: int = 0.95, 
                 top_k: int = 25, 
                 max_length: int = 64,
                 ques_per_passage: int = 1, 
                 prefix: str = "gen", 
                 batch_size: int = 32,
                 save: bool = True, 
                 save_after: int = 100000,
                 augment_probability: float = 0.5,
                 augment_per_query : int = 2,
                 augment_temperature: float = 0.5):
        
        self.logger.info("Starting to Generate {} Questions Per Passage using top-p (nucleus) sampling...".format(ques_per_passage))
        self.logger.info("Params: top_p = {}".format(top_p))
        self.logger.info("Params: top_k = {}".format(top_k))
        self.logger.info("Params: max_length = {}".format(max_length))
        self.logger.info("Params: ques_per_passage = {}".format(ques_per_passage))
        self.logger.info("Params: batch size = {}".format(batch_size))
        self.logger.info("Params: augment probability  = {}".format(augment_probability))
        self.logger.info("Params: augment per query  = {}".format(augment_per_query))
        self.logger.info("Params: augment temperature  = {}".format(augment_temperature))
        
        count = 0
        count_aug = 0
        corpus_ids = list(corpus.keys())
        corpus = [corpus[doc_id] for doc_id in corpus_ids]

        for start_idx in trange(0, len(corpus), batch_size, desc='pas'):            
            size = len(corpus[start_idx:start_idx + batch_size])
            queries = self.model.generate(
                corpus=corpus[start_idx:start_idx + batch_size], 
                ques_per_passage=ques_per_passage,
                max_length=max_length,
                top_p=top_p,
                top_k=top_k
                )
            
            assert len(queries) == size * ques_per_passage
            # For each query create augment_per_query augmented queries
            
            if self.augment_model:
                augmented = self.augment_model.augment(queries, 
                augment_per_query = augment_per_query, 
                top_p = top_p, 
                top_k = max_length, 
                max_length = max_length,
                temperature = augment_temperature)
                assert len(augmented) == size * ques_per_passage * augment_per_query
                
                augmented_query = {}
                start = 0
                for id in range(len(queries)):
                    key_query = queries[id].strip()
                    augmented_query[key_query] = augmented[start : start + augment_per_query]
                    start += augment_per_query
                
            for idx in range(size):      
                # Saving generated questions after every "save_after" corpus ids
                if (len(self.queries) % save_after == 0 and len(self.queries) >= save_after):
                    self.logger.info("Saving {} Generated Queries...".format(len(self.queries)))
                    self.save(output_dir, self.queries, self.qrels, prefix)

                # Get the corpus ids no need to change
                corpus_id = corpus_ids[start_idx + idx]
                # Each corpus has ques_per_passage number of queries
                start_id = idx * ques_per_passage
                end_id = start_id + ques_per_passage
                # Get the generated queries for the corpus.
                query_set = set([q.strip() for q in queries[start_id:end_id]])
                # For each generated query in the corpus, add it to the query set.
                for query in query_set:
                    count += 1
                    query_id = "genQ" + str(count)
                    self.queries[query_id] = query
                    self.qrels[query_id] = {corpus_id: 1}
                    # Save each augmented query as related to this corpus, and map augmented query to query
                    if self.augment_model:
                        for i in range(augment_per_query):
                            count_aug += 1
                            augmented_query_id = "augQ" + str(count_aug)
                            self.queries[augmented_query_id] = augmented_query[query][i]
                            self.qrels[augmented_query_id] = {corpus_id: 1}
                            self.aug_to_q[augmented_query_id] = query_id
                    

        
        # Saving finally all the questions
        self.logger.info("Saving {} Generated Queries...".format(len(self.queries)))
        self.save(output_dir, self.queries, self.qrels, prefix)
        self.save_query_aug(output_dir, self.aug_to_q, prefix)
    