from gpl_improved.hard_negative_miner import HardNegativeWriter
from gpl_improved.query_models import QueryAugmentMod
from gpl_improved.utils import SCORE,load_dataset,BEIR_DATASETS
from gpl_improved.query_models import QueryAugmentMod
import torch, gc
from typing import List
import hydra
from omegaconf import DictConfig

def dataset(dataset_name: BEIR_DATASETS,
         output_folder: str):
    return load_dataset(dataset_name=dataset_name, output_folder=output_folder)

def hard_negative_miner(path:str, 
                        negatives_per_query: int, 
                        query_augment_mod: QueryAugmentMod,
                        models: List[str], 
                        score : List[SCORE],
                        use_train_qrels: bool,
                        prefix : str):
    miner = HardNegativeWriter(negatives_per_query=negatives_per_query, path_to_data= path, gpl_data_prefix=prefix, query_augment_mod= query_augment_mod)
    miner.generate(models = models, score = score, use_train_qrels = use_train_qrels)
    

@hydra.main(version_base = None, config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:    
    if cfg.data.given_path == "":
        path = dataset(dataset_name=BEIR_DATASETS(cfg.data.dataset_name), output_folder=cfg.data.output_folder)
    else:
        path = cfg.data.given_path
        
    hard_negative_miner(path = path,
                        negatives_per_query= cfg.hard_negative_miner.negatives_per_query, 
                        query_augment_mod=QueryAugmentMod(cfg.query_writer.augmented),
                        models=cfg.hard_negative_miner.models, 
                        score=[SCORE(score) for score in cfg.hard_negative_miner.score],
                        use_train_qrels=cfg.hard_negative_miner.use_train_qrels,
                        prefix = cfg.data.prefix)
    
    
if __name__ == "__main__":
    gc.collect()
    torch.cuda.empty_cache()
    main()
    gc.collect()
    torch.cuda.empty_cache()
