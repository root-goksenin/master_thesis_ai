
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
@click.option("--split", type = str, help = "Path to BEIR formatted data")
def shot(model_name,data_path,split):
    # Always zero-shot the test data
    data_name = os.path.split(data_path)[1]
    output_dir = os.path.join(
                "./zero_shot_results", f"{data_name}", f"{model_name}"
            )
    if not os.path.exists(os.path.join(output_dir, "results_query_level.json")):
        corpus, queries, qrels = GenericDataLoader(data_path).load(split)
        # Load a sentence bert model, model_name needs to be loadable from the sentence bert repo.
        model = load_sbert(
                model_name,
                pooling=None,
                max_seq_length=350,
            )
        # Evaluate the sentence bert model from queries and corpus
        evaluator = EvaluateGPL(model, queries, corpus, score_function=SCORE.DOT if "cos" not in model_name else SCORE.COS)
        # Write the evaluated results
        evals = RetriverWriter(
                evaluator,
                output_dir=output_dir, 
                write_scores=True
            )
        evals.evaluate_query_based(qrels)  
    # else:
    #     print(f"Path Exsists :{output_dir}")


if __name__ == '__main__':
    gc.collect()
    torch.cuda.empty_cache()
    shot()
    gc.collect()
    torch.cuda.empty_cache()
