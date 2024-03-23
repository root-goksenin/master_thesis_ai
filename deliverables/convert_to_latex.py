import json
import os
import click
from functools import partial

def rounded(func):
    def wrapee(*args):
        args = [round(arg, 1) for arg in args]
        return func(*args)
    return wrapee

def return_row(s1,s2,s3,s4, s5):
    return fr"& {s1} & {s2} & {s3} & {s4}    &  {s5}   \\ \midrule"

@rounded
def return_last_row(s1,s2,s3,s4, s5):
    return fr"& {s1} & {s2} & {s3} & {s4}    &  {s5}   \\ \bottomrule"
    
def get_row(data_name, data, func):
    if data_name == "cqadupstack":
        return mean_of_cqadupstack(data, func)
    bm25_scores = data["BM25"][data_name][func]
    msmarco_scores = data["msmarco-distilbert-dot-v5"][data_name][func]
    multiqa_scores = data["multi-qa-distilbert-dot-v1"][data_name][func]
    gpl_baseline_scores = data["GPL Baseline"][data_name][func]
    gpl_scores = data["GPL"][data_name][func]

    return return_row(bm25_scores, gpl_baseline_scores, gpl_scores ,msmarco_scores, multiqa_scores)


def mean_of_cqadupstack(data, func):
    data_names = os.listdir("../beir_data/cqadupstack")
    bm25_scores = sum(
        data["BM25"][data_name][func] for data_name in data_names
    ) / len(data_names)
    msmarco_scores = sum(
        data["msmarco-distilbert-dot-v5"][data_name][func]
        for data_name in data_names
    ) / len(data_names)
    multiqa_scores = sum(
        data["multi-qa-distilbert-dot-v1"][data_name][func]
        for data_name in data_names
    ) / len(data_names)
    gpl_baseline_scores = sum(
        data["GPL Baseline"][data_name][func] for data_name in data_names
    ) / len(data_names)
    gpl_scores = sum(
        data["GPL"][data_name][func] for data_name in data_names
    ) / len(data_names)
    return return_last_row(bm25_scores, gpl_baseline_scores, gpl_scores, msmarco_scores, multiqa_scores)

begin = r"""\begin{table}[]
\centering
\begin{tabular}{@{}l|l|l|l|l|l|l|@{}}
\toprule
& BM25 & GPL Baseline & GPL & MSMARCO & MultiQA \\ \midrule
"""


@click.command()
@click.argument('input', nargs=1)
@click.argument('func', nargs=1)
def main(input, func):
    with open(input, "r") as file:
        data = json.load(file)
    get_row_ = partial(get_row, data = data, func = func)
    beir_table = fr"""NF-Corpus      {get_row_('nfcorpus')}
    FIQA           {get_row_('fiqa')}
    SCIDocs        {get_row_('scidocs')}
    FEVER          {get_row_('fever')}
    Arguana        {get_row_('arguana')}
    SCIFACT        {get_row_('scifact')}
    TREC-COVID     {get_row_('trec-covid')}
    Climate-FEVER  {get_row_('climate-fever')}
    HotPot-QA      {get_row_('hotpotqa')}
    NQ             {get_row_('nq')}
    Quora          {get_row_('quora')}
    WebisTouche    {get_row_('webis-touche2020')}
    DBPedia-Entity {get_row_('dbpedia-entity')}
    CQADupstack    {get_row_('cqadupstack')}
    """
    
    lotte_test_table = fr"""
    Lifestyle      {get_row_('lifestyle_search_test')}
    Technology     {get_row_('technology_search_test')}
    Writing        {get_row_('writing_search_test')}
    Recreation     {get_row_('recreation_search_test')}
    Science        {get_row_('science_search_test')}
    Pooled         {get_row_('pooled_search_test')}
    Lifestyle      {get_row_('lifestyle_forum_test')}
    Technology     {get_row_('technology_forum_test')}
    Writing        {get_row_('writing_forum_test')}
    Recreation     {get_row_('recreation_forum_test')}
    Science        {get_row_('science_forum_test')}
    Pooled         {get_row_('pooled_forum_test')}
    """
    
    lotte_dev_table = fr"""
    Lifestyle      {get_row_('lifestyle_search_dev')}
    Technology     {get_row_('technology_search_dev')}
    Writing        {get_row_('writing_search_dev')}
    Recreation     {get_row_('recreation_search_dev')}
    Science        {get_row_('science_search_dev')}
    Pooled         {get_row_('pooled_search_dev')}
    Lifestyle      {get_row_('lifestyle_forum_dev')}
    Technology     {get_row_('technology_forum_dev')}
    Writing        {get_row_('writing_forum_dev')}
    Recreation     {get_row_('recreation_forum_dev')}
    Science        {get_row_('science_forum_dev')}
    Pooled         {get_row_('pooled_forum_dev')}
    """
    
    end = f"""\end{"{tabular}"}
    \caption{"{" + func + "}"}
    \label{"{tab:my-table}"}
    \end{"{table}"}
    """
    beir_table = begin + beir_table + end
    lotte_test_table_ = begin + lotte_test_table + end
    lotte_dev_table_ = begin + lotte_dev_table + end

    with open(f"tables/beir_table_{func}.txt", 'w') as f:
        f.write(beir_table)

    with open(f"tables/lotte_test_table_{func}.txt", 'w') as f:
        f.write(lotte_test_table_)

    with open(f"tables/lotte_dev_table_{func}.txt", 'w') as f:
        f.write(lotte_dev_table_)

if __name__ == "__main__":
    main()
    