import logging
import os
from typing import List

from gpl_improved.utils import SCORE
from gpl_improved.hard_negative_miner import NegativeMiner
from gpl_improved.query_models import QueryAugmentMod


class HardNegativeWriter:

  def __init__(self, negatives_per_query: int, path_to_data: str, gpl_data_prefix: str= "gpl", 
              query_augment_mod:QueryAugmentMod = QueryAugmentMod.UsePast ):
    self.path_to_data = path_to_data
    self.gpl_data_prefix = gpl_data_prefix
    self.negatives_per_query = negatives_per_query
    self.logger = logging.getLogger(__name__ + ".HardNegativeWriter")
    self.mod = query_augment_mod
  def generate(self, models: List[str], score : List[SCORE], use_train_qrels: bool = False, remine = False, out_path_for_remine = None):
    #### Hard-negative mining ####
    #### This will be skipped if there is an existing `hard-negatives.jsonl` file under `path_to_generated_data` ####
    if remine:
      self.logger.info("No hard-negative data found. Now mining it")
      miner = NegativeMiner(
          self.path_to_data,
          self.gpl_data_prefix,
          retrievers=models,
          retriever_score_functions= list(map(lambda x: x.value, score)),
          nneg=self.negatives_per_query,
          use_train_qrels=use_train_qrels,
          query_augment_mod = self.mod,
          out_path = None, 
      )
      miner.run_with_pretrained(models[0], score[0].value, out_path = out_path_for_remine)
    elif ("hard-negatives.jsonl" in os.listdir(self.path_to_data)): 
      self.logger.info("Using exisiting hard-negative data")
    else:
      self.logger.info("No hard-negative data found. Now mining it")
      miner = NegativeMiner(
          self.path_to_data,
          self.gpl_data_prefix,
          retrievers=models,
          retriever_score_functions= list(map(lambda x: x.value, score)),
          nneg=self.negatives_per_query,
          use_train_qrels=use_train_qrels,
          query_augment_mod = self.mod,
          out_path = "hard-negatives.jsonl"
      )
      miner.run()