
import time
import os
import torch
import json
import logging
import numpy as np
import tqdm

from beir.datasets.data_loader import GenericDataLoader
from gpl_improved.query_models import QueryAugmentMod
from sentence_transformers import SentenceTransformer
from easy_elasticsearch import ElasticSearchBM25


def parse_jsonl_file(file_path):
    augq_mapping = {}

    with open(file_path, 'r') as file:
        for line in file:
            json_data = json.loads(line)
            genq_value = json_data['text']
            augq_value = json_data['_id']

            if augq_value not in augq_mapping:
                augq_mapping[augq_value] = None

            augq_mapping[augq_value] = genq_value

    return augq_mapping

class NegativeMiner(object):
    def __init__(
        self,
        generated_path,
        prefix,
        retrievers=["bm25", "msmarco-distilbert-base-v3", "msmarco-MiniLM-L-6-v3"],
        retriever_score_functions=["none", "cos_sim", "cos_sim"],
        nneg=50,
        use_train_qrels: bool = False,
        query_augment_mod: QueryAugmentMod = QueryAugmentMod.UsePast,
        out_path = "hard-negatives.jsonl"
    ):
        self.logger = logging.getLogger(__name__ + ".NegativeMiner")
        if use_train_qrels:
            self.logger.info("Using labeled qrels to construct the hard-negative data")
            self.corpus, self.gen_queries, self.gen_qrels = GenericDataLoader(
                generated_path
            ).load(split="train")
        else:
            self.corpus, self.gen_queries, self.gen_qrels = GenericDataLoader(
                generated_path, prefix=prefix
            ).load(split="train")
        self.output_path = os.path.join(generated_path, out_path)
        self.retrievers = retrievers
        self.retriever_score_functions = retriever_score_functions
        if "bm25" in retrievers:
            assert (
                nneg <= 10000
            ), "Only `negatives_per_query` <= 10000 is acceptable by Elasticsearch-BM25"
            assert retriever_score_functions[retrievers.index("bm25")] == "none"

        assert set(retriever_score_functions).issubset({"none", "dot", "cos_sim"})

        self.nneg = nneg
        if nneg > len(self.corpus):
            self.logger.warning(
                "`negatives_per_query` > corpus size. Please use a smaller `negatives_per_query`"
            )
            self.nneg = len(self.corpus)
        self.query_augment_mod = query_augment_mod
        self.generated_path = generated_path

    def _get_doc(self, did):
        return " ".join([self.corpus[did]["title"], self.corpus[did]["text"]])

    def _mine_sbert(self, model_name, score_function, pre_trained = False):
        assert score_function in ["dot", "cos_sim"]
        normalize_embeddings = False
        if score_function == "cos_sim":
            normalize_embeddings = True

        result = {}
        sbert = model_name if pre_trained else SentenceTransformer(model_name)
        docs = list(map(self._get_doc, self.corpus.keys()))
        dids = np.array(list(self.corpus.keys()))
        doc_embs = sbert.encode(
            docs,
            batch_size=128,
            show_progress_bar=True,
            convert_to_numpy=False,
            convert_to_tensor=True,
            normalize_embeddings=normalize_embeddings,
        )
        qids = list(self.gen_qrels.keys())
        queries = list(map(lambda qid: self.gen_queries[qid], qids))
        for start in tqdm.trange(0, len(queries), 128):
            qid_batch = qids[start : start + 128]
            qemb_batch = sbert.encode(
                queries[start : start + 128],
                show_progress_bar=False,
                convert_to_numpy=False,
                convert_to_tensor=True,
                normalize_embeddings=normalize_embeddings,
            )
            score_mtrx = torch.matmul(qemb_batch, doc_embs.t())  # (qsize, dsize)
            _, indices_topk = score_mtrx.topk(k=self.nneg + 1, dim=-1)
            indices_topk = indices_topk.tolist()
            for qid, neg_dids in zip(qid_batch, indices_topk):
                  neg_dids = dids[neg_dids].tolist()
                  for pos_did in self.gen_qrels[qid]:
                      if pos_did in neg_dids:
                          neg_dids.remove(pos_did)
                  result[qid] = neg_dids[:self.nneg]


        if (self.query_augment_mod == QueryAugmentMod.UsePast):
            # Load the query -> aug pair.
            # Overwrite the result[qid] of aug with real query.
            print("Using real queries as ground truths for augmented ones")
            genq_mapping = parse_jsonl_file(os.path.join(self.generated_path, "gpl-aug.jsonl"))
            new_result = {}
            for qid,neg_dids in result.items():
              if "aug" in qid:
                # Get the real query from corresponding aug
                real_query = genq_mapping[qid]
                # The aug query has same results as real query
                new_result[qid] = result[real_query]
              else:
                # If it is already result, just put neg_dids.
                new_result[qid] = neg_dids
            result = new_result
        elif (self.query_augment_mod == QueryAugmentMod.None_):
            print("Not using augmented queries")
            new_result = {
                qid: neg_dids
                for qid, neg_dids in result.items()
                if "aug" not in qid
            }
            result = new_result
        else:
            print("Retriving results for the augmented queries")       


        print("Retriving")
        return result

    def _mine_bm25(self):
        self.logger.info(f"Mining with bm25")
        result = {}
        docs = list(map(self._get_doc, self.corpus.keys()))
        dids = np.array(list(self.corpus.keys()))
        pool = dict(zip(dids, docs))
        bm25 = ElasticSearchBM25(
            pool,
            port_http="9222",
            port_tcp="9333",
            service_type="executable",
            index_name=f"one_trial{int(time.time() * 1000000)}",
        )
        for qid, pos_dids in tqdm.tqdm(self.gen_qrels.items()):
            query = self.gen_queries[qid]
            rank = bm25.query(query, topk=self.nneg)  # topk should be <= 10000
            neg_dids = list(rank.keys())
            for pos_did in self.gen_qrels[qid]:
                if pos_did in neg_dids:
                    neg_dids.remove(pos_did)
            result[qid] = neg_dids
        return result

    def run(self):
        hard_negatives = {}
        for retriever, score_function in zip(
            self.retrievers, self.retriever_score_functions
        ):
            if retriever == "bm25":
                hard_negatives[retriever] = self._mine_bm25()
            else:
                hard_negatives[retriever] = self._mine_sbert(model_name = retriever, score_function = score_function)

        self.logger.info("Combining all the data")
        result_jsonl = []
        for qid, pos_dids in tqdm.tqdm(self.gen_qrels.items()):
            line = {
                "qid": qid,
                "pos": list(pos_dids.keys()),
                "neg": {k: v[qid] for k, v in hard_negatives.items()},
            }
            result_jsonl.append(line)

        self.logger.info(f"Saving data to {self.output_path}")
        with open(self.output_path, "w") as f:
            for line in result_jsonl:
                f.write(json.dumps(line) + "\n")
        self.logger.info("Done")
    
    def run_with_pretrained(self, model, score_function, out_path):
        hard_negatives = {}
        # Hacky to get it, but works. 
        hard_negatives["pre_trained"] = self._mine_sbert(model_name = model, score_function= score_function, pre_trained = True)

        self.logger.info("Combining all the data")
        result_jsonl = []
        for qid, pos_dids in tqdm.tqdm(self.gen_qrels.items()):
            line = {
                "qid": qid,
                "pos": list(pos_dids.keys()),
                "neg": {k: v[qid] for k, v in hard_negatives.items()},
            }
            result_jsonl.append(line)

        self.logger.info(f"Saving data to {out_path}")
        with open(out_path, "w") as f:
            for line in result_jsonl:
                f.write(json.dumps(line) + "\n")
        self.logger.info("Done")
    