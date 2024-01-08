
from beir import util, LoggingHandler
from enum import Enum
import os
from torch.utils.data import Dataset
import json
from sentence_transformers.readers.InputExample import InputExample
import random
import linecache
from typing import Dict
import logging
from sentence_transformers import SentenceTransformer, models
import logging
from torch import Tensor

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
