
import click 
import os
from beir.datasets.data_loader import GenericDataLoader
from gpl_improved.trainer.RetriverWriter import EvaluateGPL, RetriverWriter
from gpl_improved.utils import load_sbert
from gpl_improved.utils import SCORE
import logging 
import gc 
import torch

logging.basicConfig(level = logging.INFO)

@click.command()
@click.option("--model_name", type = str, help = "Zero Shot Dense Retriver model to use")
@click.option("--data_path", type = str, help = "Path to BEIR formatted data")
def shot(model_name,data_path):
    # Always zero-shot the test data
    corpus, queries, qrels = GenericDataLoader(data_path).load("test")
    # Load a sentence bert model, model_name needs to be loadable from the sentence bert repo.
    model = load_sbert(
            model_name,
            pooling=None,
            max_seq_length=350,
        )
    # Evaluate the sentence bert model from queries and corpus
    evaluator = EvaluateGPL(model, queries, corpus, score_function=SCORE.DOT if "cos" not in model_name else SCORE.COS)
    data_name = os.path.split(data_path)[1]
    # Write the evaluated results
    evals = RetriverWriter(
            evaluator,
            output_dir=os.path.join(
                "./zero_shot_results", f"{data_name}", f"{model_name}"
            ),
        )
    evals.evaluate_beir_format(qrels)  


if __name__ == '__main__':
    gc.collect()
    torch.cuda.empty_cache()
    shot()
    gc.collect()
    torch.cuda.empty_cache()
