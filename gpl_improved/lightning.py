from sentence_transformers import CrossEncoder
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import transformers
import logging
from torch.optim import AdamW
import os 
from typing import List
from gpl_improved.trainer.loss import MarginDistillationLoss
from gpl_improved.trainer.hard_negative_dataset import HardNegativeDataset, hard_negative_collate_fn
from gpl_improved.utils import load_sbert, batch_to_device
from gpl_improved.trainer.RetriverWriter import EvaluateGPL, RetriverWriter, BM25Wrapper
from sentence_transformers.readers.InputExample import InputExample
import torch
from beir.datasets.data_loader import GenericDataLoader
import pytorch_lightning as pl
from torch.cuda.amp import autocast
from gpl_improved.query_models import QueryAugmentMod


def reduce_teacher_labels(labels, mode):
    if mode == "average":
        return avg_teacher_labels(labels)
    elif mode == "normalize":
        return avg_teacher_labels(normalized_teacher_labels(labels))
    else:
        return labels
        

def avg_teacher_labels(labels):
    '''
    Labels in shape B x T 
    T being the teacher number
    '''
    return labels.mean(dim = 1)


def normalized_teacher_labels(labels):
    '''
    Labels in shape B x T 
    T being the teacher number
    '''
    return labels / labels.std(dim = 0)


class GPLDistill(pl.LightningModule):
    def __init__(self, cross_encoder, bi_retriver, path, 
                 eval_every = 1000,  batch_size: int = 32, warmup_steps = 1000, t_total = 140000, amp_training = True,
                 evaluate_baseline = True, max_seq_length = 350, reducer = "mean", query_per_passage = 3, augmented_mod = QueryAugmentMod.None_, 
                 save_name = "", prefix = "", bm25_reweight = True, bm25_weight = 2,  corpus_name = "fiqa"):
        super().__init__()
        self.save_hyperparameters()
        self.logger_ = logging.getLogger(f".GPLDistill{__name__}")
        self.logger_.info(f"Evaluating the model every {eval_every} step")
        self.logger_.info(f"Using Cross Encoders {cross_encoder}")
        self.logger_.info(f"Adapting bi_retriver {bi_retriver}")
        self.logger_.info(f"Using batch size {batch_size}")
        self.logger_.info(f"Warming up the LR scheduler with {warmup_steps} step")
        self.logger_.info(f"Training the model total of {t_total} steps")
        self.logger_.info(f"Using AMP training {amp_training}")
        self.logger_.info(f"Evaluate Baseline {evaluate_baseline}")
        self.logger_.info(f"Using maximum sequence length of {max_seq_length}")
        self.logger_.info(f"Using BM25 reweighting {bm25_reweight}")

        self.bm25_reweight = bm25_reweight
        self.bm25_weight = bm25_weight
        self.corpus_name = corpus_name
        self.prefix = prefix
        self.save_name = save_name
        self.eval_every = eval_every
        self.path= path
        self.evaluate_baseline = evaluate_baseline
        self.reducer = reducer
        self.query_per_passage = query_per_passage
        self.augment_mod = augmented_mod
        
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.base_model = bi_retriver
        self.max_seq_length = max_seq_length
        self.bi_retriver = load_sbert(bi_retriver, pooling = None, max_seq_length = self.max_seq_length)
        self.teacher_names = cross_encoder
        self.teachers = [CrossEncoder(cross_encoder_) for cross_encoder_ in cross_encoder]
        self.teacher_models = [teacher.model for teacher in self.teachers]
        self.retokenizers = [AutoTokenizer.from_pretrained(cross_encoder_) for cross_encoder_ in cross_encoder]
        self.loss = MarginDistillationLoss(model=self.bi_retriver, similarity_fct="dot")
        self.batch_size = batch_size
        self.max_grad_norm = 1
        self.use_amp = amp_training
        self.automatic_optimization = False
        
        self.logger_.info(f"Gradient clipping of norm {self.max_grad_norm}")
        if self.use_amp:
            self.logger_.info("Using Mixed Predicision")
            self.scaler = torch.cuda.amp.GradScaler()

        self.bi_retriver.zero_grad()
        self.bi_retriver.train()

    def setup(self,stage):
        corpus, queries, qrels = GenericDataLoader(self.path, prefix=self.prefix).load(split="train")
        corpus_test, queries_test, qrels_test = GenericDataLoader(self.path).load(split="test")

        self.train_queries = queries
        self.train_corpus = corpus 
        self.train_qrels = qrels
        self.test_queries = queries_test
        self.test_corpus = corpus_test
        self.test_qrels = qrels_test

        if self.evaluate_baseline:
            self.eval_test()
        
    def on_train_end(self):
        '''
        When training finishes, evaluate test set.
        '''
        self.eval_test()
        
        
    def add_teacher_statistics(self, labels):
        means = {f'mean_{teacher}': labels.mean(dim = 1)[id] for id,teacher in enumerate(self.teacher_names)}
        stds = {f'mean_{teacher}': labels.std(dim = 0)[id] for id,teacher in enumerate(self.teacher_names)}
        self.logger.experiment.add_scalars("Means of teachers", 
                                            means, 
                                            global_step=self.global_step)
        self.logger.experiment.add_scalars("STD of teachers",
                                           stds,
                                            global_step=self.global_step)
    def training_step(self, batch, batch_idx):
        '''
        Take a training step with the batch. Batch contains the query generated by T5, positive document and negative document.
        '''
        if (self.global_step) % self.eval_every == 0:
          try:
            ndcgs_train = self.ndcg_train(k_values = [10,])
            self.log("ndcg_train", ndcgs_train["NDCG@10"])
          except RuntimeError:
            self.log("ndcg_train", 0)
          try:
            ndcgs_test = self.ndcg_test(k_values = [10,])        
            self.log("ndcg_test", ndcgs_test["NDCG@10"])
          except RuntimeError:
            self.log("ndcg_test", 0)
        
        skip_scheduler = False
        bi_optimizer, cross_optimizer, = self.optimizers()
        bi_scheduler, cross_scheduler = self.lr_schedulers()
        _ , (query, pos, neg) = batch

        # We do not need teacher gradients.
        with torch.no_grad():
            teacher_labels = torch.zeros((self.batch_size, len(self.teacher_models)),device = self.device)
            # Here we get the distillation labels from cross encoder. For [query, pos] and [query,neg] pairs
            for i, (cross_encoder, retokenizer) in enumerate(zip(self.teachers, self.retokenizers)):
                query, pos, neg = [self.retokenize(texts, retokenizer) for texts in [query, pos, neg]]
                scores = cross_encoder.predict(
                    list(zip(query, pos)) + list(zip(query, neg)), show_progress_bar=False, convert_to_tensor = True
                )
                teacher_labels[:, i] = scores[: len(query)] - scores[len(query) :]

            self.add_teacher_statistics(teacher_labels)
            teacher_labels = reduce_teacher_labels(teacher_labels, self.reducer)
        train_examples = [InputExample(texts=[q, p, n], label=label) for q,p,n,label in zip(query,pos,neg,teacher_labels)]
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
        my_lr = bi_scheduler.get_last_lr()
        self.log("learning_rate", my_lr[0])
        self.log("Distill_loss", loss_value)

    def ndcg_train(self, k_values: List[int] = None):
        """
        Calculate the NDCG scores for training data.
        Args:
            k_values: A list of integers representing the cutoff values for evaluation. Default is [10].

        Returns:
            ndcg: A dictionary containing the NDCG scores for different cutoff values.
        """
        if k_values is None:
            k_values = [10]
        retriver = EvaluateGPL(self.bi_retriver, self.train_queries, self.train_corpus)
        return retriver.evaluate(self.train_qrels, k_values = k_values)
    
    def ndcg_test(self, k_values: List[int] = None):
        """
        Calculate the NDCG scores for test data.

        Args:
            k_values: A list of integers representing the cutoff values for evaluation. Default is [10].

        Returns:
            ndcg: A dictionary containing the NDCG scores for different cutoff values.
        """
        if k_values is None:
            k_values = [10]
        retriver = EvaluateGPL(self.bi_retriver, self.test_queries, self.test_corpus)
        return retriver.evaluate(self.test_qrels, k_values = k_values)
    
    def configure_optimizers(self):
        optimizer_bi = AdamW(self.bi_retriver.parameters(), lr=2e-5, weight_decay = 0.01)
        optimizer_cross = AdamW(self.teacher_models[0].parameters(), lr=2e-5, weight_decay = 0.01)
        scheduler_bi = transformers.get_linear_schedule_with_warmup(optimizer_bi, num_warmup_steps=self.warmup_steps, num_training_steps=self.t_total)
        scheduler_cross = transformers.get_linear_schedule_with_warmup(optimizer_cross, num_warmup_steps=self.warmup_steps, num_training_steps=self.t_total)
        
        self.logger_.info("Using AdamW optimizer for bi retriver with weight decay of 0.01, and learning rate of 2e-5")
        self.logger_.info("Using AdamW optimizer for cross encoder with weight decay of 0.01, and learning rate of 2e-5")
        return [optimizer_bi, optimizer_cross], [scheduler_bi, scheduler_cross]
    
    
    def train_dataloader(self):
        corpus, gen_queries, _ = GenericDataLoader(self.path, prefix=self.prefix).load(split="train")
        hard_negative_dataset = HardNegativeDataset(
        os.path.join(self.path, "hard-negatives.jsonl"), gen_queries, corpus
        )
        hard_negative_dataloader = DataLoader(
        hard_negative_dataset, shuffle=True, batch_size=self.batch_size, drop_last=True,
        )
        hard_negative_dataloader.collate_fn = hard_negative_collate_fn
        return hard_negative_dataloader

    def eval_test(self):
        self.bi_retriver.eval()        
        self._evaluate("test", '_GPL_test')
        self.bi_retriver.train()
        self.bi_retriver.zero_grad()
    def _evaluate(self, split, arg1):
        corpus, queries, qrels = GenericDataLoader(self.path).load(split=split)
        retriver = EvaluateGPL(self.bi_retriver, queries, corpus)
        if self.bm25_reweight:
            self.logger_.info("Using the BM25 Wrapper for evaluator")
            retriver = BM25Wrapper(retriver, bm25_reweight = self.bm25_reweight, corpus_name = self.corpus_name, bm25_weight = self.bm25_weight)
        writer = RetriverWriter(
            retriver=retriver,
            output_dir=os.path.join(
                self.path,
                f"{self.base_model}{arg1}",
                f"test_step_{self.global_step}",
                f"nr_queries_per_passage_{self.query_per_passage}",
                f"augmented_mod_{self.augment_mod}_{self.save_name}",
                f"bm25_reweight={self.bm25_reweight}"
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
