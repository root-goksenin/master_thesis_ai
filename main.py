from gpl_improved.hard_negative_miner import HardNegativeWriter
from gpl_improved.query_models import QueryAugmentMod,QueryWriter
from gpl_improved.trainer.hard_negative_dataset import HardNegativeDataset, hard_negative_collate_fn
from gpl_improved.utils import load_dataset, BEIR_DATASETS, SCORE
from gpl_improved.query_models import QueryAugmentMod
from gpl_improved.lightning import GPLDistill
from beir.datasets.data_loader import GenericDataLoader
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger
import os 
import logging 
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
import torch, gc
from typing import List
import hydra
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint

def dataset(dataset_name: BEIR_DATASETS,
         output_folder: str):
    return load_dataset(dataset_name=dataset_name, output_folder=output_folder)

def query_writer(path:str,
         queries_per_passage: int,
         batch_size: int,
         augmented: QueryAugmentMod,
         use_train_qrels: bool,
         top_p : float,
         top_k : int,
         max_length: int,
         augment_probability: float,
         forward_model_path: str,
         back_model_path: str,
         augment_per_query: int,
         augment_temperature:float,
         prefix = str) -> int:
    
    writer = QueryWriter(queries_per_passage= queries_per_passage, path_to_data= path, gpl_data_prefix=prefix)
    writer.generate(use_train_qrels=use_train_qrels, 
                    batch_size = batch_size, 
                    augmented = augmented,
                    top_p = top_p, 
                    top_k = top_k,
                    max_length = max_length,
                    augment_probability= augment_probability,
                    forward_model_path=forward_model_path,
                    back_model_path = back_model_path,
                    augment_per_query= augment_per_query,
                    augment_temperature= augment_temperature)
    return writer.queries_per_passage

def hard_negative_miner(path:str, 
                        negatives_per_query: int, 
                        query_augment_mod: QueryAugmentMod,
                        models: List[str], 
                        score : List[SCORE],
                        use_train_qrels: bool,
                        prefix : str):
    miner = HardNegativeWriter(negatives_per_query=negatives_per_query, path_to_data= path, gpl_data_prefix=prefix, query_augment_mod= query_augment_mod)
    miner.generate(models = models, score = score, use_train_qrels = use_train_qrels)
    
def train(path : str,
             cross_encoders: List[str],
             bi_retriver: str,
             t_total: int,
             eval_every: int,
             batch_size: int,
             warmup_steps: int,
             amp_training : bool,
             evaluate_baseline: bool,
             max_seq_length: int,
             seed: int,
             name: str,
             q_per_passage: int, 
             augmented_mod: QueryAugmentMod,
             prefix: str,
             reducer: str,
             bm25_reweight: bool,
             corpus_name: str,
             bm25_weight: int,
             ):
    logger = TensorBoardLogger("tb_logs", name=name)
    # 140,000 steps for every BEIR dataset.
    checkpoint_callback = ModelCheckpoint(monitor='ndcg_train',
                                          dirpath=f'./saved_models/gpl_improved/{name}',
                                          mode='max',
                                          filename = '{step}-{ndcg_train:.3f}',
                                          verbose = True,
                                          save_on_train_epoch_end = True,
                                          save_last = True)
    trainer = pl.Trainer(logger = logger, gpus = 1, max_epochs = -1, max_steps = t_total, deterministic = True, callbacks = [checkpoint_callback])
    seed_everything(seed, workers=True)
    # Train the distillation
    # Batch size is 32.    
    ## We can have multiple cross-encoders to distill the knowledge from.
    distill = GPLDistill(cross_encoder= cross_encoders,
                         bi_retriver = bi_retriver, 
                         path = path,
                         amp_training = amp_training, 
                         batch_size = batch_size, 
                         evaluate_baseline=evaluate_baseline,
                         eval_every=eval_every,
                         warmup_steps=warmup_steps,
                         max_seq_length=max_seq_length,
                         query_per_passage=q_per_passage,
                         augmented_mod=augmented_mod,
                         save_name = name,
                         prefix = prefix,
                         reducer = reducer,
                         bm25_reweight= bm25_reweight,
                         corpus_name = corpus_name,
                         bm25_weight= bm25_weight,
                         )
    trainer.fit(model=distill)
    

@hydra.main(version_base = None)
def main(cfg: DictConfig) -> None:
    print(cfg.data.given_path)
    print(cfg.trainer.name)
    if cfg.data.given_path == "":
        path = dataset(dataset_name=BEIR_DATASETS(cfg.data.dataset_name), output_folder=cfg.data.output_folder)
    else:
        path = cfg.data.given_path
    resolved_queries_per_passage = query_writer(path = path, 
                queries_per_passage = cfg.query_writer.queries_per_passage, 
                batch_size = cfg.query_writer.batch_size,
                augmented=QueryAugmentMod(cfg.query_writer.augmented), 
                use_train_qrels=cfg.query_writer.use_train_qrels, 
                top_p = cfg.query_writer.top_p , 
                top_k = cfg.query_writer.top_k, 
                max_length = cfg.query_writer.max_length,
                augment_probability = cfg.query_writer.augment_probability, 
                forward_model_path = cfg.query_writer.forward_model_path, 
                back_model_path = cfg.query_writer.back_model_path, 
                augment_per_query = cfg.query_writer.augment_per_query, 
                augment_temperature = cfg.query_writer.augment_temperature,
                prefix = cfg.data.prefix)

    hard_negative_miner(path = path,
                        negatives_per_query= cfg.hard_negative_miner.negatives_per_query, 
                        query_augment_mod=QueryAugmentMod(cfg.query_writer.augmented),
                        models=cfg.hard_negative_miner.models, 
                        score=[SCORE(score) for score in cfg.hard_negative_miner.score],
                        use_train_qrels=cfg.hard_negative_miner.use_train_qrels,
                        prefix = cfg.data.prefix)
    
    train(path = path, 
            cross_encoders = cfg.trainer.cross_encoders, 
            bi_retriver = cfg.trainer.bi_retriver, 
            t_total = cfg.trainer.t_total, 
            eval_every = cfg.trainer.eval_every, 
            batch_size = cfg.trainer.batch_size, 
            warmup_steps = cfg.trainer.warmup_steps, 
            amp_training = cfg.trainer.amp_training,
            evaluate_baseline = cfg.trainer.evaluate_baseline, 
            max_seq_length = cfg.trainer.max_seq_length, 
            seed = cfg.trainer.seed,
            name = cfg.trainer.name,
            q_per_passage= resolved_queries_per_passage,
            augmented_mod= QueryAugmentMod(cfg.query_writer.augmented),
            prefix = cfg.data.prefix,
            reducer = cfg.trainer.reducer,
            bm25_reweight = cfg.beir_evaluator.bm25_reweight,
            corpus_name = cfg.beir_evaluator.corpus_name,
            bm25_weight = cfg.beir_evaluator.bm25_weight
            )
    
if __name__ == "__main__":
    gc.collect()
    torch.cuda.empty_cache()
    main()
    gc.collect()
    torch.cuda.empty_cache()
