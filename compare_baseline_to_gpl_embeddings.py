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
from collections import namedtuple

topk_namedtuple = namedtuple('topk_namedtuple', ['values', 'indices'])
def topk(array: np.ndarray, k: int, largest: bool = True) -> topk_namedtuple:
    """Returns the k largest/smallest elements and corresponding indices 
    from an array-like input.
    Parameters
    ----------
    array : np.ndarray or list
        the array-like input
    k : int
        the k in "top-k" 
    largest ： bool, optional
        controls whether to return largest or smallest elements        
    Returns
    -------
    namedtuple[values, indices]
        Returns the :attr:`k` largest/smallest elements and corresponding indices 
        of the given :attr:`array`
    Example
    -------
    >>> array = [5, 3, 7, 2, 1]
    >>> topk(array, 2)
    >>> topk_namedtuple(values=array([7, 5]), indices=array([2, 0], dtype=int64))
    >>> topk(array, 2, largest=False)
    >>> topk_namedtuple(values=array([1, 2]), indices=array([4, 3], dtype=int64))
    >>> array = [[1, 2], [3, 4], [5, 6]]
    >>> topk(array, 2)
    >>> topk_namedtuple(values=array([6, 5]), indices=(array([2, 2], dtype=int64), array([1, 0], dtype=int64)))
    """

    array = np.asarray(array)
    flat = array.ravel()

    if largest:
        indices = np.argpartition(flat, -k)[-k:]
        argsort = np.argsort(-flat[indices])
    else:
        indices = np.argpartition(flat, k)[:k]
        argsort = np.argsort(flat[indices])

    indices = indices[argsort]
    values = flat[indices]
    indices = np.unravel_index(indices, array.shape)
    if len(indices) == 1:
        indices, = indices
    return values, indices

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
    print(find_most_disimilar(query_mapping, query_sim))
    print("CORPUS:")
    print(find_most_similar(doc_mapping, doc_sim, k = 3))
    print(find_most_disimilar(doc_mapping, doc_sim, k = 3))

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

def find_most_similar(query_id_mapping, query_sim, k = 10):
    values, indices = topk(query_sim, k)
    return [query_id_mapping[id] for id in indices[0]], values

def find_most_disimilar(query_id_mapping, query_sim, k = 10):
    values, indices = topk(query_sim, k, False)
    return [query_id_mapping[id] for id in indices[0]], values

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




