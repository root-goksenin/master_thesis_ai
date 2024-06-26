
from enum import Enum
import os
from typing import Dict
import logging
from sentence_transformers import SentenceTransformer, models
from torch import Tensor
import torch
import tqdm
from beir import util
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict

class BaseSearch(ABC):

    @abstractmethod
    def search(self, 
               corpus: Dict[str, Dict[str, str]], 
               queries: Dict[str, str], 
               top_k: int, 
               **kwargs) -> Dict[str, Dict[str, float]]:
        pass

def reweight_results(result_dense, result_bm25, weight = 0.1):
    for q_id in tqdm.tqdm(result_dense.keys()): 
        # Get the document scores for query with q_id
        dense_score = result_dense[q_id]
        bm25_score = result_bm25[q_id]
        result_bm25[q_id] = {key: dense_score[key] + (weight * bm25_score[key]) for key in bm25_score.keys()}
    return result_bm25

def load_pretrained_bi_retriver(data_name: str, model_name: str = "gpl_average", aug_strategy: str = "no_aug", checkpoint_name : str = "last"):
    model = torch.load(
        f"/home/gyuksel/master_thesis_ai/saved_models/gpl_improved/{data_name}_{aug_strategy}_{model_name}/{checkpoint_name}.ckpt",
        map_location=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),
    )
    bi_retriver_weights = {
        ".".join(layer.split(".")[1:]): weight
        for layer, weight in model["state_dict"].items()
        if "bi_retriver" in layer
    }
    bi_retriver = load_sbert(
        "GPL/msmarco-distilbert-margin-mse", pooling=None, max_seq_length=350
    )
    bi_retriver.load_state_dict(bi_retriver_weights)
    return bi_retriver


def batch_to_device(batch, target_device):
    """
    send a pytorch batch to a device (CPU/GPU)
    """
    for key in batch:
        if isinstance(batch[key], Tensor):
            batch[key] = batch[key].to(target_device)
    return batch


class BEIR_DATASETS(Enum):
    ARGUANA = "arguana"
    CLIMATE_FEVER = "climate-fever"
    CQADUP = "cqadupstack"
    DBPEDIA = "dbpedia-entity"
    FEVER = "fever"
    FIQA = "fiqa"
    GERMAN = "germanquad"
    HOTPOT = "hotpotqa"
    MMARCO = "mmarco"
    mrtydi = "mrtydi"
    MSMARCO_V2 = "msmarco-v2"
    MSMARCO = "msmarco"
    NF = "nfcorpus"
    NQ_TRAIN = "nq-train"
    NQ = "nq"
    QUORA = "quora"
    SCIDOCS = "scidocs"
    SCIFACT = "scifact"
    TREC_COVID_BEIR = "trec-covid-beir"
    TREC_COVID_V2 = "trec-covid-v2"
    TREC_COVID = "trec-covid"
    VIHEALTHQA = "vihealthqa"
    WEBIST = "webis-touche2020"

class SCORE(Enum):
   COS = "cos_sim"
   DOT = "dot"

url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip"

def load_dataset(dataset_name: BEIR_DATASETS, output_folder: str):
  resolved_url = url.format(dataset_name.value)
  data_path = os.path.join(output_folder, dataset_name.value)
  # Do not download a dataset that had been downloaded before
  if not os.path.exists(data_path):
    util.download_and_unzip(resolved_url, output_folder)
  return data_path


logger = logging.getLogger(__name__)


def directly_loadable_by_sbert(model: SentenceTransformer):
    loadable_by_sbert = True
    try:
        texts = [
            "This is an input text",
        ]
        model.encode(texts)
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            raise e
        else:
            loadable_by_sbert = False
    return loadable_by_sbert


def load_sbert(model_name_or_path, pooling=None, max_seq_length=None):
    model = SentenceTransformer(model_name_or_path)

    ## Check whether SBERT can load the checkpoint and use it
    ## Loadable by SBERT directly
    ## Mainly two cases: (1) The checkpoint is in SBERT-format (e.g. "bert-base-nli-mean-tokens"); (2) it is in HF-format but the last layer can provide a hidden state for each token (e.g. "bert-base-uncased")
    ## NOTICE: Even for (2), there might be some checkpoints (e.g. "princeton-nlp/sup-simcse-bert-base-uncased") that uses a linear layer on top of the CLS token embedding to get the final dense representation. In this case, setting `--pooling` to a specify pooling method will misuse the checkpoint. This is why we recommend to use SBERT-format if possible
    ## Setting pooling if needed
    if pooling is not None:
        logger.warning(
            f"Trying setting pooling method manually (`--pooling={pooling}`). Recommand to use a checkpoint in SBERT-format and leave the `--pooling=None`: This is less likely to misuse the pooling"
        )
        last_layer: models.Pooling = model[-1]
        assert (
            type(last_layer) == models.Pooling
        ), f"The last layer is not a pooling layer and thus `--pooling={pooling}` cannot work. Please try leaving `--pooling=None` as in the default setting"
        # We here change the pooling by building the whole SBERT model again, which is safer and more maintainable than setting the attributes of the Pooling module
        word_embedding_model = models.Transformer(
            model_name_or_path, max_seq_length=max_seq_length
        )
        pooling_model = models.Pooling(
            word_embedding_model.get_word_embedding_dimension(),
            pooling_mode=pooling,
        )
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    if max_seq_length is not None:
        first_layer: models.Transformer = model[0]
        assert (
            type(first_layer) == models.Transformer
        ), "Unknown error, please report this"
        assert hasattr(
            first_layer, "max_seq_length"
        ), "Unknown error, please report this"
        setattr(
            first_layer, "max_seq_length", max_seq_length
        )  # Set the maximum-sequence length
        logger.info(f"Set max_seq_length={max_seq_length}")

    logger.info("Finished loading the sentence transformer model")
    return model

def get_word_idx(sent: str, word: str):
    return sent.split(" ").index(word)


def get_hidden_states(encoded, token_ids_word, model, layers):
    """Push input IDs through model. Stack and sum `layers` (last four by default).
    Select only those subword token outputs that belong to our word of interest
    and average them."""
    with torch.no_grad():
        output = model(**encoded)

    # Get all hidden states
    states = output.hidden_states
    # Stack and sum all requested layers
    output = torch.stack([states[i] for i in layers]).sum(0).squeeze()
    # Only select the tokens that constitute the requested word
    word_tokens_output = output[token_ids_word]

    return word_tokens_output.mean(dim=0)


def get_sentence_embeddings(sent, model):
    """Get a word vector by first tokenizing the input sentence, getting all token idxs
    that make up the word of interest, and then `get_hidden_states`."""
    sentence_embedding = model.encode(sent, convert_to_numpy= True, convert_to_tensor = False)
    if len(sentence_embedding.shape) == 1:
        sentence_embedding = np.expand_dims(sentence_embedding, axis = 0)
    return sentence_embedding


def get_token_embeddings(sent, model):
    token_embeddings = model.encode(sent, output_value = "token_embeddings").cpu().numpy()
    if len(token_embeddings.shape) == 1:
        token_embeddings = np.expand_dims(token_embeddings, axis = 0)
    return token_embeddings
   


