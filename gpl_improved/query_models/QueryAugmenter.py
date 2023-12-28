from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch, logging
from typing import List, Union
import numpy as np
from gpl_improved.query_models import generate

class QAugmentModel:
    def __init__(self, forward_model_path: str, back_model_path: str, gen_prefix: str = "", use_fast: bool = True, device: str = None, **kwargs):
        self.forward_tokenizer = AutoTokenizer.from_pretrained(forward_model_path, use_fast=use_fast)
        self.forward_model = AutoModelForSeq2SeqLM.from_pretrained(forward_model_path)
        self.back_tokenizer = AutoTokenizer.from_pretrained(back_model_path, use_fast=use_fast)
        self.back_model = AutoModelForSeq2SeqLM.from_pretrained(back_model_path)
        self.logger = logging.getLogger(__name__ + ".QAugmentModel")

        self.gen_prefix = gen_prefix
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info("Use pytorch device: {}".format(self.device))
        self.forward_model = self.forward_model.to(self.device)
        self.back_model = self.back_model.to(self.device)

    def augment(self, queries: List[str],  augment_per_query: int, top_k: int, max_length: int, top_p: float = None, temperature: float = None, probability : float = 1) -> Union[List[str], None]:

        # Augmentation probability.
        probabilities = np.random.uniform(0,1, (len(queries)))
        forward_texts = [(query) for id, query in enumerate(queries) if probabilities[id] < probability]

        if len(forward_texts) > 0:
          encodings = self.forward_tokenizer(forward_texts, padding=True, truncation=True, return_tensors="pt")
          # Top-p nucleus sampling
          # https://huggingface.co/blog/how-to-generate
          outs = generate(model = self.forward_model, encodings = encodings, device = self.device,
                          augment_per_query = augment_per_query, top_k = top_k, max_length = max_length, top_p = top_p, temperature = temperature)
          backward_texts = self.forward_tokenizer.batch_decode(outs, skip_special_tokens=True)
          encodings = self.back_tokenizer(backward_texts, padding=True, truncation=True, return_tensors="pt")
          # Top-p nucleus sampling
          # https://huggingface.co/blog/how-to-generate
          outs = generate(model = self.back_model, encodings = encodings, device = self.device,
                          augment_per_query = 1, top_k = top_k, max_length = max_length, top_p = top_p, temperature = temperature)
          augmented_queries = self.back_tokenizer.batch_decode(outs, skip_special_tokens=True)
          return augmented_queries
        else:
          None
