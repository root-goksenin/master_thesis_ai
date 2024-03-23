from gpl_improved.query_models import QueryAugmentMod,QueryWriter
from gpl_improved.utils import load_dataset, BEIR_DATASETS
from gpl_improved.query_models import QueryAugmentMod
import torch, gc
import hydra
from omegaconf import DictConfig


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


@hydra.main(version_base = None, config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    if cfg.data.given_path == "":
        path = dataset(dataset_name=BEIR_DATASETS(cfg.data.dataset_name), output_folder=cfg.data.output_folder)
    else:
        path = cfg.data.given_path
    
    query_writer(path = path, 
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

if __name__ == "__main__":
    gc.collect()
    torch.cuda.empty_cache()
    main()
    gc.collect()
    torch.cuda.empty_cache()
