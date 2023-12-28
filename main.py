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
from gpl_improved.trainer.trainer import PseudoLabeler
from gpl_improved.utils import load_dataset, BEIR_DATASETS, SCORE
from gpl_improved.query_models import QueryAugmentMod
from beir.datasets.data_loader import GenericDataLoader
import logging 

logger = logging.getLogger("__main__")


if __name__ == "__main__":
    # Load any BEIR dataset to the specified directory
    path = load_dataset(BEIR_DATASETS.FIQA, "./generated_improved")
    # Generate queries using the T5 model.
    writer = QueryWriter(queries_per_passage= 5, path_to_data= path, gpl_data_prefix="imp_gpl")
    # This is where the probabilitiy, and augmentation comes in. We augmented using the augmented keyword
    writer.generate(use_train_qrels=False, batch_size = 128, augmented = QueryAugmentMod.None_)
    # From the generated queries, and augmented queries, retrievev hard negatives. If query_augment_mod is new, retrieve hard negatives for augmented queries also
    miner = HardNegativeWriter(negatives_per_query=50, path_to_data= path, gpl_data_prefix="imp_gpl", query_augment_mod= QueryAugmentMod.None_)
    # Generate using the models.
    miner.generate(models = ["msmarco-distilbert-base-v3", "msmarco-MiniLM-L-6-v3"], score = [SCORE.COS, SCORE.COS], use_train_qrels=False)
    logger = TensorBoardLogger("tb_logs", name="gpl_model_try")
    trainer = pl.Trainer(logger = logger, gpus = 1, max_epochs = 1, min_steps = 10000,deterministic = True )
    cross_encoders=["cross-encoder/ms-marco-MiniLM-L-6-v2"]
    bi_retriver="multi-qa-mpnet-base-dot-v1"
    seed_everything(42, workers=True)
    distill = GPLDistill(cross_encoders[0], bi_retriver, path = path, amp_training = True)
    trainer.fit(model=distill)
    gc.collect()
    torch.cuda.empty_cache()
    corpus, gen_queries, _ = GenericDataLoader(path, prefix="imp_gpl").load(split="train")
    # Start training with the labeller. 
    trainer = PseudoLabeler(generated_path=path, gen_queries=gen_queries, corpus=corpus, 
                            total_steps=18000, batch_size= 32, cross_encoder="cross-encoder/ms-marco-MiniLM-L-6-v2",
                            max_seq_length = 350, gpl_score_function= SCORE.DOT, base_model="multi-qa-mpnet-base-dot-v1", eval_every= 6000, eval_dir=path)
    trainer.train()
