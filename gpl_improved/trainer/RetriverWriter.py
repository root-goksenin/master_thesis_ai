import logging 
import os 
import json
import numpy as np 
from typing import List 
from gpl_improved.utils import SCORE, reweight_results
from beir.retrieval import models
from beir.retrieval.search.lexical import BM25Search as BM25
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
import pytrec_eval

class RetriverWriter:
    def __init__(self, retriver, output_dir):
        self.retriver = retriver
        self.k_values = [1, 3, 5, 10, 20, 100]
        self.output_dir = output_dir
        self.logger = logging.getLogger(f"{__name__}.RetriverWriter")

    def evaluate_beir_format(self, query_corpus_relativity):
        '''
        Evaluate the retriver results using the BEIR format. We copy the code from GPL.
        This is how GPL evaluates their results.
        '''
        self.logger.info("Evaluating retrieved results")
        
        ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(
            query_corpus_relativity, self.retriver.results(), self.k_values
        )
        mrr = EvaluateRetrieval.evaluate_custom(query_corpus_relativity, self.retriver.results(), self.k_values, metric="mrr")
        ndcg, _map, recall, precision, mrr = self.calculate_metrics(ndcg, _map, recall, precision, mrr)
        result_path = self.write_to_dir(ndcg, _map, recall, precision, mrr)
        self.logger.info(f"Saved evaluation results to {result_path}")

    def calculate_metrics(self, ndcg, _map, recall, precision, mrr):
        ndcgs = [ndcg]
        _maps = [_map]
        recalls = [recall]
        precisions = [precision]
        mrrs = [mrr]
        # We have a database scheme for each query has ndcg values at different cut-off points.
        ndcg = {k: np.mean([score[k] for score in ndcgs]) for k in ndcg}
        _map = {k: np.mean([score[k] for score in _maps]) for k in _map}
        recall = {k: np.mean([score[k] for score in recalls]) for k in recall}
        precision = {k: np.mean([score[k] for score in precisions]) for k in precision}
        mrr = {k: np.mean([score[k] for score in mrrs]) for k in mrr}
        return ndcg,_map,recall,precision,mrr

    def write_to_dir(self, ndcg, _map, recall, precision, mrr):
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
            
        return result_path

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
        try:
          new_scores[int(query) - 1 ] = {}
          for key, value in scores[query].items():
            new_scores[int(query) - 1][key.replace("_cut_", "@")] = value
        # We may get a value error when query is not integer!
        except ValueError as e:
          new_scores[query] = {}
          for key, value in scores[query].items():
            new_scores[query][key.replace("_cut_", "@")] = value        

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
  """
  A class for evaluating retrieval results using the BEIR format.

  Args:
      model: The model to be evaluated.
      query: The queries.
      corpus: The corpus.
      k_values: A list of integers representing the cutoff values for evaluation. Default is [1, 3, 5, 10, 20, 100].
      score_function: The scoring function to be used for evaluation. Default is SCORE.DOT.

  Attributes:
      retriever: The retriever object.
      logger: The logger object.
      k_values: A list of integers representing the cutoff values for evaluation.
      query: The query.
      corpus: The corpus.
      results_: The retrieval results.

  Methods:
      evaluate: Evaluate the retrieval results and return only ndcg and mrrs for logging.
      results: Retrieve the results.

  """

  def __init__(self, model, query, corpus, score_function: SCORE = SCORE.DOT):
      # Model to be Evaluated. This is gonna be transformed into self.qmodel and self.dmodel from sentence bert
      self.retriever = models.SentenceBERT(sep=" ")
      self.retriever.q_model = model
      self.retriever.doc_model = model
      self.logger = logging.getLogger(f"{__name__}.EvaluateGPL")
      model_dres = DRES(self.retriever, batch_size=32)
      self.query = query
      self.corpus = corpus
      self.retriever = EvaluateRetrieval(
          model_dres, score_function=score_function.value, k_values=[len(self.corpus)]

      )
      self.results_ = None

  def evaluate(self, qrels, k_values):
      ndcg, _, _, _ = EvaluateRetrieval.evaluate(
          qrels, self.results(), k_values
      )
      ndcgs = [ndcg]
      # We have a database scheme for each query has ndcg values at different cut-off points.
      ndcg = {k: np.mean([score[k] for score in ndcgs]) for k in ndcg}
      return ndcg  
    
  def results(self):
      if self.results_ is None:
          self.logger.info(" Initializing retirever model and retriving corpus, queries")
          results_ = self.retriever.retrieve(self.corpus, self.query)
          self.results_ = results_
      return self.results_
    
    
    
class BM25Wrapper():
  """
    A class that wraps a retriever and provides additional functionality.

    Args:
        wrapped_retriver: The wrapped retriever object.
        bm25_reweight: Whether to apply BM25 reweighting. Default is False.
        eval_bm25: Whether to evaluate BM25. Default is False.
        corpus_name: The name of the corpus. Default is "scifact".
        bm25_weight: The weight for BM25. Default is 0.1.

    Attributes:
        wrapper: The wrapped retriever object.
        bm25_reweight: Whether to apply BM25 reweighting.
        eval_bm25: Whether to evaluate BM25.
        logger: The logger object.
        bm25_results: The BM25 results.
        bm25_weight: The weight for BM25.
        k_values: The k values.

    Raises:
        FileNotFoundError: If the BM25 scores file is not found.

    Examples:
        >>> retriever = RetrieverWriter(wrapped_retriver, bm25_reweight=True)
        >>> retriever.evaluate_bm25()
  """
  def __init__(self, wrapped_retriver, bm25_reweight = False, eval_bm25 = False, corpus_name = "scifact", bm25_weight = 0.1):
    self.wrapper = wrapped_retriver
    self.bm25_reweight = bm25_reweight
    self.eval_bm25 = eval_bm25
    self.logger = logging.getLogger(f"{__name__}.BM25Wrapper")
    self.results_ = None
    if self.bm25_reweight or eval_bm25:
      with open(f"/home/gyuksel/master_thesis_ai/bm25_scores/{corpus_name}/bm25_scores.json", 'r') as f:
        self.bm25_results = json.load(f)
        self.bm25_weight = bm25_weight
        # k_values needs to be the length of corpus if we want to reweight everything :)
        self.k_values = [len(self.wrapper.corpus)]

  def results(self):
    if self.results_ is None:
      if self.bm25_reweight:
        self.logger.info(f"Re-weighting with BM25 scores with weight={self.bm25_weight}")
        results_ = reweight_results(self.wrapper.results(), self.bm25_results, weight = self.bm25_weight)
        self.results_ = results_
      elif self.eval_bm25:
        self.logger.info("Evaluating BM25")
        self.results_ = self.bm25_results
      else:
        self.results_ = self.wrapper.results()
    return self.results_
    