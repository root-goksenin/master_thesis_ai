from gpl_improved.utils import load_sbert, load_pretrained_bi_retriver
from beir.datasets.data_loader import GenericDataLoader
import click
import os
import logging
import numpy as np
from beir.retrieval import models
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from beir.retrieval.evaluation import EvaluateRetrieval
import matplotlib.pyplot as plt

def vectorized_cos_sim(a_array, b_array):
    norm_a = np.linalg.norm(a_array, axis=1, keepdims=True)
    norm_b = np.linalg.norm(b_array, axis=1, keepdims=True)
    dot_product = np.sum(a_array * b_array, axis=1, keepdims=True)
    return dot_product / (norm_a * norm_b)

logging.basicConfig(level=logging.DEBUG)

@click.command()
@click.option('--data_name', type=click.Choice(os.listdir('./bm25_scores')), help='The name of the data')
@click.option('--aug_strategy', type=click.Choice(['no_aug', 'aug', 'use_past', 'aug_None']), help='The augmentation strategy')
@click.option('--model_name', type=click.Choice(['mini_lm', 'tiny_bert', 'mini_lm_average','tiny_bert_average', 'two_teacher_average', 'two_teacher_normalized', 'three_teacher_average', 'three_teacher_normalized']), help='The name of the model')
def main(data_name, aug_strategy, model_name):
    bi_retriver_after_gpl = load_pretrained_bi_retriver(data_name=data_name, model_name=model_name, aug_strategy=aug_strategy)
    bi_retriver_before_gpl = load_sbert("GPL/msmarco-distilbert-margin-mse", pooling=None, max_seq_length=350)
    data_path = f"/home/gyuksel/master_thesis_ai/gpl_given_data/{data_name}"
    test_corpus, test_queries, _ = GenericDataLoader(data_path).load(split="test")
    doc_sim, doc_mapping  = compute_embedding_sim(bi_retriver_before_gpl, bi_retriver_after_gpl, test_corpus, encode_func='encode_corpus')
    query_sim, query_mapping = compute_embedding_sim(bi_retriver_before_gpl, bi_retriver_after_gpl, test_queries, encode_func='encode_queries')
    before_results, adapted_results = compute_retrival_results(bi_retriver_before_gpl, bi_retriver_after_gpl, test_queries, test_corpus)
    
    plot_comparison(test_corpus, doc_sim, query_sim, before_results, adapted_results)
    print(find_most_similar(query_mapping, query_sim))
    # find_most_dissimilar(test_corpus, doc_sim)

def plot_comparison(test_corpus, doc_sim, query_sim, before_results, adapted_results):
    avg_before = np.zeros((len(before_results), len(test_corpus)))
    avg_adapted = np.zeros_like(avg_before)
    for id, q_id in enumerate(before_results):
        avg_before[id] = np.array(list(before_results[q_id].values()))
        avg_adapted[id] = np.array(list(adapted_results[q_id].values()))

    fig = plt.figure()
    plt.hist(avg_before.mean(axis = 0), label="before")
    plt.hist(avg_adapted.mean(axis = 0), label="adapted")
    plt.legend()
    plt.savefig("before_after_hist.png")
    fig = plt.figure()
    plt.hist(doc_sim)
    plt.savefig("doc_sim.png")
    fig = plt.figure()
    plt.hist(query_sim)
    plt.savefig("query_sim.png")

def find_most_similar(query_id_mapping, query_sim):
    sorted_indices = np.argsort(query_sim)[::-1]
    # Get top 10 indices
    top_10_indices = sorted_indices[:10]
    return [query_id_mapping[id] for id in top_10_indices]

def compute_embedding_sim(model_baseline, model_after, data, encode_func):
    data = [data[d_id] for d_id in data]
    mapping = []
    for data_ in data:
        if "queries" in encode_func:
            mapping.append(data_)
        else:
            mapping.append(data_['title'] + data_['text'])
    model_baseline_bert = build_retrieval_model(model_baseline)
    model_adapted_bert = build_retrieval_model(model_after)
    embeddings_before = getattr(model_baseline_bert, encode_func)(data, batch_size=32, show_progress_bar=True, convert_to_tensor=False)
    embeddings_after = getattr(model_adapted_bert, encode_func)(data, batch_size=32, show_progress_bar=True, convert_to_tensor=False)
    return vectorized_cos_sim(embeddings_before, embeddings_after), mapping

def build_retrieval_model(model):
    result = models.SentenceBERT(sep=" ")
    result.q_model = model
    result.doc_model = model
    return result

def compute_retrival_results(model_baseline, model_after, queries, corpus):
    before_results = EvaluateRetrieval(DRES(build_retrieval_model(model_baseline),  batch_size=32),score_function="dot", k_values=[len(corpus)]).retrieve(corpus, queries)
    after_results = EvaluateRetrieval(DRES(build_retrieval_model(model_after), batch_size=32),score_function="dot", k_values=[len(corpus)]).retrieve(corpus, queries)
    return before_results, after_results

    

if __name__ == '__main__':
    main()




