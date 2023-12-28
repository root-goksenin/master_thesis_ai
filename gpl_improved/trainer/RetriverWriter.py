import random
import logging 
import os 
import json
import numpy as np 
from typing import List 
from gpl_improved.utils import SCORE
from beir.retrieval import models
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
import pytrec_eval


class RetriverWriter:
    def __init__(self, retriver, output_dir):
      # Retriver needs to have results
      # Retriver needs to have k_values.
      self.retriver = retriver
      self.k_values = retriver.k_values
      self.output_dir = output_dir
      self.logger = logging.getLogger(__name__ + ".RetriverWriter")

    def evaluate_beir_format(self, query_corpus_relativity):
      '''
      Evaluate the retriver results using the BEIR format. We copy the code from GPL.
      This is how GPL evaluates their results.
      '''
      # Evaluate retrieved results
      #### Evaluate your retrieval using NDCG@k, MAP@K ...

      ndcgs = []
      _maps = []
      recalls = []
      precisions = []
      mrrs = []
      self.logger.info(f"Evaluating retrieved results")
      ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(
          query_corpus_relativity, self.retriver.results(), self.k_values
      )
      mrr = EvaluateRetrieval.evaluate_custom(query_corpus_relativity, self.retriver.results(), self.k_values, metric="mrr")

      ndcgs.append(ndcg)
      _maps.append(_map)
      recalls.append(recall)
      precisions.append(precision)
      mrrs.append(mrr)
      # We have a database scheme for each query has ndcg values at different cut-off points.
      ndcg = {k: np.mean([score[k] for score in ndcgs]) for k in ndcg}
      _map = {k: np.mean([score[k] for score in _maps]) for k in _map}
      recall = {k: np.mean([score[k] for score in recalls]) for k in recall}
      precision = {k: np.mean([score[k] for score in precisions]) for k in precision}
      mrr = {k: np.mean([score[k] for score in mrrs]) for k in mrr}

      os.makedirs(self.output_dir, exist_ok=True)
      result_path = os.path.join(self.output_dir, "results.json")
      with open(result_path, "w") as f:
          json.dump(
              {
                  "ndcg": ndcg,
                  "map": _map,
                  "recall": recall,
                  "precicion": precision,
                  "mrr": mrr,
              },
              f,
              indent=4,
          )
      self.logger.info(f"Saved evaluation results to {result_path}")

    def evaluate_query_based(self, query_corpus_relativity):
      '''
      Evaluate Retriver using query based evaluations.

      '''
      map_string = "map_cut." + ",".join([str(k) for k in self.k_values])
      ndcg_string = "ndcg_cut." + ",".join([str(k) for k in self.k_values])
      recall_string = "recall." + ",".join([str(k) for k in self.k_values])
      precision_string = "P." + ",".join([str(k) for k in self.k_values])
      evaluator = pytrec_eval.RelevanceEvaluator(query_corpus_relativity, {map_string, ndcg_string, recall_string, precision_string})
      scores = evaluator.evaluate(self.retriver.results())
      new_scores = {}
      for query in scores:
        new_scores[int(query) - 1 ] = {}
        for key, value in scores[query].items():
          new_scores[int(query) - 1][key.replace("_cut_", "@")] = value

      os.makedirs(self.output_dir, exist_ok=True)
      result_path = os.path.join(self.output_dir, "results_query_level.json")
      with open(result_path, "w") as f:
          json.dump(
              new_scores,
              f,
              indent=4,
          )
      self.logger.info(f"Saved evaluation results to {result_path}")



class EvaluateGPL:
  def __init__(self, model, query, corpus, k_values: List[int] = [1, 3, 5, 10, 20, 100], score_function: SCORE = SCORE.DOT) :
    # Model to be Evaluated. This is gonna be transformed into self.qmodel and self.dmodel from sentence bert
    self.retriever = models.SentenceBERT(sep=" ")
    self.retriever.q_model = model
    self.retriever.doc_model = model
    self.logger = logging.getLogger(__name__ + ".EvaluateGPL")
    model_dres = DRES(self.retriever, batch_size=16)
    # Score_function has to be dot or cos_sim. This is used for scoring the similarity.
    # Wrap self.retriever into evaluate_retriever class

    # We use self.k_values here to retrieve top_k.
    self.retriever = EvaluateRetrieval(
        model_dres, score_function=score_function.value, k_values=k_values

    )
    self.query = query
    self.corpus = corpus

    # To calculate ndcg at @K
    self.k_values = k_values
    self.results_ = None

  def evaluate(self, qrels, k_values):
      ndcgs = []
      mrrs = []
      ndcg, _, _, _ = EvaluateRetrieval.evaluate(
          qrels, self.results(), k_values
      )
      mrr = EvaluateRetrieval.evaluate_custom(qrels, self.results(), k_values, metric="mrr")

      ndcgs.append(ndcg)
      mrrs.append(mrr)
      # We have a database scheme for each query has ndcg values at different cut-off points.
      ndcg = {k: np.mean([score[k] for score in ndcgs]) for k in ndcg}
      mrr = {k: np.mean([score[k] for score in mrrs]) for k in mrr}
      return ndcg, mrr

  def results(self,):
        # First retrieve using retriever and get b
      if  self.results_ is None:
          self.logger.info(" Initializing retirever model and retriving corpus, queries")
          results_ = self.retriever.retrieve(self.corpus, self.query)
          self.results_ = results_
      return self.results_