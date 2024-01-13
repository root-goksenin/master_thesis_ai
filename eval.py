
from gpl_improved.trainer.RetriverWriter import RetriverWriter, EvaluateGPL, BM25Wrapper
from gpl_improved.utils import load_sbert, load_pretrained_bi_retriver
from beir.datasets.data_loader import GenericDataLoader
import click
import os
import logging

# Configure the logging module
logging.basicConfig(level=logging.DEBUG)

@click.command()
@click.option('--data_name', type=click.Choice(os.listdir('./bm25_scores')), help='The name of the data')
@click.option('--aug_strategy', type=click.Choice(['no_aug', 'aug', 'use_past']), help='The augmentation strategy')
@click.option('--model_name', type=click.Choice(['mini_lm', 'tiny_bert', 'two_teacher_average', 'two_teacher_normalized', 'three_teacher_average', 'three_teacher_normalized']), help='The name of the model')
@click.option('--use_bm25', is_flag=True, help='Whether to use BM25 reweighting')
@click.option('--bm25_weight', type=float, default=2.0, help='The weight for BM25')
@click.option('--load_pretrained', is_flag=True, help='Whether to load a pretrained model')
def main(data_name, aug_strategy, model_name, use_bm25, bm25_weight, load_pretrained):
    """
    A command-line interface for evaluating retrieval models.
    """
    if load_pretrained and (aug_strategy is None or model_name is None):
        raise click.UsageError("Both --aug_strategy and --model_name must be specified when --load_pretrained is used.")
    
    corpus, queries, qrels = GenericDataLoader(f"./gpl_given_data/{data_name}").load(split="test")
    bi_retriver = (
        load_pretrained_bi_retriver(
            data_name=data_name,
            model_name=model_name,
            aug_strategy=aug_strategy,
        )
        if load_pretrained
        else load_sbert(
            "GPL/msmarco-distilbert-margin-mse",
            pooling=None,
            max_seq_length=350,
        )
    )
    retriver = EvaluateGPL(bi_retriver,queries, corpus)
    if use_bm25:
        bm25_weight = bm25_weight
        retriver = BM25Wrapper(retriver, use_bm25, corpus_name=data_name, bm25_weight = bm25_weight)
    evals = RetriverWriter(retriver, 
                           output_dir = 
                            os.path.join(
                            "./evaluated_models",
                            f"{data_name}",
                            f"{model_name}",
                            f"augmented_mod_{aug_strategy}",
                            f"bm25_reweight={use_bm25}_weight={bm25_weight}")
                )
    evals.evaluate_beir_format(qrels)

if __name__ == '__main__':
    main()