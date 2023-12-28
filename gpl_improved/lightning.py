from sentence_transformers import CrossEncoder
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import transformers
import tqdm 
import logging
from torch.optim import Adam, AdamW
import os 
from typing import List
from gpl_improved.trainer.loss import MarginDistillationLoss
from gpl_improved.trainer.hard_negative_dataset import HardNegativeDataset, hard_negative_collate_fn
from gpl_improved.utils import load_sbert, batch_to_device
from gpl_improved.trainer.RetriverWriter import EvaluateGPL, RetriverWriter
from sentence_transformers.readers.InputExample import InputExample
import torch
from beir.datasets.data_loader import GenericDataLoader
import matplotlib.pyplot as plt 
import numpy as np 
import pytorch_lightning as pl
from torch.optim import Adam
# define the LightningModule

class GPLDistill(pl.LightningModule):
    def __init__(self, cross_encoder, bi_retriver, path, eval_every = 5000,  batch_size: int = 32, warmup_steps = 1000, t_total = 10000, amp_training = True):
        super().__init__()
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.base_model = bi_retriver
        self.max_seq_length = 350
        self.bi_retriver = load_sbert(bi_retriver, pooling = None, max_seq_length = self.max_seq_length)
        self.cross_encoder = CrossEncoder(cross_encoder)
        self.cross_encoder_model = self.cross_encoder.model
        self.retokenizers = AutoTokenizer.from_pretrained(cross_encoder)
        self.loss = MarginDistillationLoss(model=self.bi_retriver, similarity_fct="dot")
        # This is going to be KL Divergence Loss!
        self.eval_every = eval_every
        self.path= path
        self.batch_size = batch_size
        self.max_grad_norm = 1
        self.use_amp = amp_training
        self.automatic_optimization = False

        if self.use_amp:
            from torch.cuda.amp import autocast
            print("Using amp training")
            self.scaler = torch.cuda.amp.GradScaler()
            
        self.bi_retriver.zero_grad()
        self.bi_retriver.train()
        
        # self.cross_encoder.zero_grad()
        # self.cross_encoder.train()
    def setup(self,stage):
        corpus, queries, qrels = GenericDataLoader(self.path).load(split="dev")
        self.dev_queries = queries
        self.dev_corpus = corpus 
        self.dev_qrels = qrels
      
    def training_step(self, batch, batch_idx):
        # sourcery skip: extract-method, inline-immediately-returned-variable

        # training_step defines the train loop.
        # it is independent of forward
        
        # if self.global_step % self.eval_every == 0:
        #   self.bi_retriver.eval()
        #   ndcgs, mrrs = self.ndcg_dev(k_values = [10,])
        #   self.log("ndcg@10", ndcgs["NDCG@10"])
        #   self.log("mrr@10", mrrs["MRR@10"])
        #   self.bi_retriver.train()
        #   self.bi_retriver.zero_grad()
        skip_scheduler = False
        bi_optimizer, cross_optimizer, = self.optimizers()
        bi_scheduler, cross_scheduler = self.lr_schedulers()
        _ , (query, pos, neg) = batch
        for cross_encoder, retokenizer in zip([self.cross_encoder], [self.retokenizers]):
          query, pos, neg = [self.retokenize(texts, retokenizer) for texts in [query, pos, neg]]
          scores = cross_encoder.predict(
              list(zip(query, pos)) + list(zip(query, neg)), show_progress_bar=False
          )
          labels = scores[: len(query)] - scores[len(query) :]
          labels = (
              labels.tolist()
          )  # Using `tolist` will keep more precision digits!!!

        train_examples = [InputExample(texts=[q, p, n], label=label) for q,p,n,label in zip(query,pos,neg,labels)]
        train_dataloader = DataLoader(train_examples, shuffle=False, batch_size=len(train_examples))
        train_dataloader.collate_fn = self.smart_batching_collate
        for features_, labels in train_dataloader:
          labels = labels.to(self.device)
          features = list(map(lambda batch: batch_to_device(batch, self.device), features_))
        
        
        if self.use_amp:
            with autocast():
                loss_value = self.loss(features, labels)
            scale_before_step = self.scaler.get_scale()
            self.manual_backward(self.scaler.scale(loss_value))
            self.scaler.unscale_(bi_optimizer)
            self.clip_gradients(bi_optimizer, gradient_clip_val=self.max_grad_norm, gradient_clip_algorithm="norm")
            self.scaler.step(bi_optimizer)
            self.scaler.update()
            skip_scheduler = self.scaler.get_scale() != scale_before_step
        else:
            loss_value = self.loss(features, labels)
            self.manual_backward(loss_value)
            self.clip_gradients(bi_optimizer, gradient_clip_val=self.max_grad_norm, gradient_clip_algorithm="norm")
            bi_optimizer.step()

        bi_optimizer.zero_grad()
        cross_optimizer.zero_grad()
        
        if not skip_scheduler:
            bi_scheduler.step()
            cross_scheduler.step()
        my_lr = bi_scheduler.get_lr()
        self.log("learning_rate", my_lr)
        print(my_lr)


    def configure_optimizers(self):
        optimizer_bi = AdamW(self.bi_retriver.parameters(), lr=2e-5, weight_decay = 0.01)
        optimizer_cross = AdamW(self.cross_encoder.model.parameters(), lr=2e-5, weight_decay = 0.01)
        scheduler_bi = transformers.get_linear_schedule_with_warmup(optimizer_bi, num_warmup_steps=self.warmup_steps, num_training_steps=self.t_total)
        scheduler_cross = transformers.get_linear_schedule_with_warmup(optimizer_cross, num_warmup_steps=self.warmup_steps, num_training_steps=self.t_total)

        return [optimizer_bi, optimizer_cross], [scheduler_bi, scheduler_cross]
    
    
    def train_dataloader(self):
        corpus, gen_queries, _ = GenericDataLoader(self.path, prefix="imp_gpl").load(split="train")
        hard_negative_dataset = HardNegativeDataset(
        os.path.join(self.path, "hard-negatives.jsonl"), gen_queries, corpus
        )
        hard_negative_dataloader = DataLoader(
        hard_negative_dataset, shuffle=True, batch_size=self.batch_size, drop_last=True,
        )
        hard_negative_dataloader.collate_fn = hard_negative_collate_fn
        return hard_negative_dataloader

    def ndcg_dev(self, k_values: List[int] = None):
        if k_values is None:
            k_values = [10]
        
        retriver = EvaluateGPL(self.bi_retriver, self.dev_queries, self.dev_corpus)
        return retriver.evaluate(self.dev_qrels, k_values = k_values)
      
    def eval_dev(self):
        self._evaluate("dev", '_GPL_dev')

    def eval_test(self):
        self._evaluate("test", '_GPL_test')

    def _evaluate(self, split, arg1):
        corpus, queries, qrels = GenericDataLoader(self.path).load(split=split)
        retriver = EvaluateGPL(self.bi_retriver, queries, corpus)
        writer = RetriverWriter(
            retriver=retriver,
            output_dir=os.path.join(
                self.path,
                f"{self.base_model}{arg1}",
                f"test_step_{self.global_step}",
            ),
        )
        writer.evaluate_query_based(qrels)
        writer.evaluate_beir_format(qrels)
    
    def retokenize(self, texts, retokenizer):
        ## We did this retokenization for two reasons:
        ### (1) Setting the max_seq_length;
        ### (2) We cannot simply use CrossEncoder(cross_encoder, max_length=max_seq_length),
        ##### since the max_seq_length will then be reflected on the concatenated sequence,
        ##### rather than the two sequences independently
        texts = list(map(lambda text: text.strip(), texts))
        features = retokenizer(
            texts,
            padding=True,
            truncation="longest_first",
            return_tensors="pt",
            max_length=self.max_seq_length,
        )
        return retokenizer.batch_decode(
            features["input_ids"],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
      
    def tokenize(self, texts):
        """
        Tokenizes the texts
        """
        return self.bi_retriver._first_module().tokenize(texts)

    def smart_batching_collate(self, batch):
        """
        Transforms a batch from a SmartBatchingDataset to a batch of tensors for the model
        Here, batch is a list of InputExample instances: [InputExample(...), ...]

        :param batch:
            a batch from a SmartBatchingDataset
        :return:
            a batch of tensors for the model
        """
        texts = [example.texts for example in batch]
        sentence_features = [self.tokenize(sentence) for sentence in zip(*texts)]
        labels = torch.tensor([example.label for example in batch])
        return sentence_features, labels
