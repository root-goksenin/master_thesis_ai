
import os 
from typing import List 
import json 
from nltk.tokenize import RegexpTokenizer
import tqdm
import os 
import json
import seaborn as sns 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from collections import Counter 
import scienceplots
import concurrent
from concurrent.futures import ProcessPoolExecutor
from collections import namedtuple
from typing import Dict, Tuple
import json
import os
import logging
import csv
from beir.datasets.data_loader import GenericDataLoader

vocab_tuple = namedtuple('Vocab', 'name vocab')
sns.set(font_scale=2)
plt.style.use(['science','ieee'])
memo = {}


def normalize(counter):
    total_count = sum(counter.values())
    return {key: value / total_count for key, value in counter.items()}

def get_word_freq(text_iter, queries = False):
    # Put all the words in the corpus into a list
    words = Counter()
    tokenizer = RegexpTokenizer(r'\w+')
    for text in tqdm.tqdm(text_iter.values()):
        if not queries:
            text = text['title'] + text['text'] if text['title'] != "" else text['text']
        tokenized = tokenizer.tokenize(text)
        tokenized = [w.lower() for w in tokenized]
        words.update(tokenized)

    return normalize(words)

def normalized_jaccard_similarity(vocab_1: vocab_tuple, vocab_2: vocab_tuple):
    if vocab_1.name not in memo:
        memo[vocab_1.name] = {}
        memo[vocab_1.name]
    if memo[vocab_1.name].get(vocab_2.name, None) is not None:
        return memo[vocab_1.name][vocab_2.name]
    words = set(vocab_1.vocab.keys()).union(set(vocab_2.vocab.keys()))
    up = 0
    down = 0
    for k in words:
        word_freq_1, word_freq_2 = vocab_1.vocab.get(k, 0), vocab_2.vocab.get(k,0)
        up += min(word_freq_1, word_freq_2)
        down += max(word_freq_1, word_freq_2)
    memo[vocab_1.name][vocab_2.name] = up/down
    return up/down

def plot_heatmap(df):
    plt.figure(figsize = (10,10))
    mask = np.triu(np.ones_like(df, dtype=np.bool))
    df.fillna(0)
    df = df.rename(index = {k : os.path.split(k)[1] for k in df.columns}, columns = {k : os.path.split(k)[1] for k in df.columns})
    heatmap = sns.heatmap(df, mask=mask, vmin=0, annot=True, cmap='Blues', square = False, annot_kws={"fontsize":15})
    heatmap.set_title('Corpus vocab overlap for BEIR', fontdict={'fontsize':24}, pad=16)
    heatmap.figure.savefig('jaccard_similarities_beir_try.png', bbox_inches='tight')


class CorpusComparator():
    def __init__(self, *corpus_files):
        self.corpuses: List[str] = list(corpus_files)
        print(self.corpuses)
        self.check_corpuses()
        self.load_corpuses()
    
    def check_corpuses(self):
        for corpus_file in self.corpuses:
            if not os.path.exists(os.path.join(corpus_file, "corpus.jsonl")):
                self.corpuses.remove(corpus_file)
    
    def load_corpuses(self):
        self.corpus_loaders = {}
        for corpus_file in self.corpuses:
            print(f"Loading: {corpus_file}")
            try:
                self.corpus_loaders[corpus_file] = GenericDataLoader(corpus_file).load_corpus()
            except ValueError as e:
                print(f"Got {e}, ignoring")
                return None

    def create_similarity_matrix(self, output_file: str):
        if os.path.exists(output_file):
            with open(output_file, "r") as f:
                self.similarities = json.load(f)
        else:
            self._create_similarity_matrix(output_file)
        self.plot_similarity_matrix()

    def _create_similarity_matrix(self, output_file):
        vocabs = {
            data: get_word_freq(loader)
            for data, loader in self.corpus_loaders.items()
        }
        self.similarities = {}
        for key_1, vocab_1 in vocabs.items():
            for key_2, vocab_2 in vocabs.items():
                if key_1 != key_2: 
                    if key_1 not in self.similarities:
                        self.similarities[key_1] = []
                    data_1, data_2 = vocab_tuple(key_1, vocab_1), vocab_tuple(key_2, vocab_2)
                    self.similarities[key_1].append({key_2: normalized_jaccard_similarity(data_1, data_2)})
        
        with open(output_file, 'w') as writer:
            json.dump(self.similarities, writer)
        
        return self.similarities
    def plot_similarity_matrix(self):
        row_column_names = list(self.similarities.keys())
        square_df = pd.DataFrame(index=row_column_names, columns=row_column_names)
        for row in row_column_names:
            for col in row_column_names:
                if row == col:
                    square_df.loc[row, col] = 1.0
                else:
                    square_df.loc[row, col] = [dict_val[col] for dict_val in self.similarities[row] if col in dict_val][0]
        plot_heatmap(square_df)




def get_word_freq_from_text(text):
    words = Counter()
    tokenizer = RegexpTokenizer(r'\w+')
    tokenized = tokenizer.tokenize(text)
    words = Counter(tokenized)
    return normalize(words)
      
def get_avg_overlap(path):
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        data = json.load(f)
    m = 0
    total = 0
    for v in data.values():
        for v_ in v.values():
            m += v_
            total += 1
    return m / total


def calculate_overlap(text_1, text_2):
    word_freq_1 = get_word_freq_from_text(text_1)
    word_freq_2 = get_word_freq_from_text(text_2) 
    return normalized_jaccard_similarity(word_freq_1, word_freq_2)

class QRelsComparator():
    def __init__(self, *corpus_files):
        self.corpuses: List[str] = list(corpus_files)
        self.check_corpuses()
        self.load_corpuses()
    
    def check_corpuses(self):
        for corpus_file in self.corpuses:
            if not os.path.exists(os.path.join(corpus_file, "corpus.jsonl")):
                self.corpuses.remove(corpus_file)
    
    def load_corpuses(self, splits = None):
        if splits is None:
            splits = ["test" for _ in range(len(self.corpuses))]
        self.corpus_loaders = {}
        for corpus_file, split in zip(self.corpuses, splits):
            print(f"Loading: {corpus_file}")
            try:
                self.corpus_loaders[corpus_file] = GenericDataLoader(corpus_file).load(split)
            except ValueError as e:
                print(f"Got {e}, ignoring")
                return None

    
    def check_query_overlap(self, output_file: str = None):
        # Check the query vocab overlap between msmarco, and every other beir format dataset
        splits = ["train" if corpus == "msmarco" else "test" for corpus in self.corpuses]
        self.load_corpuses(splits)
        vocabs = {
            key: get_word_freq(queries, queries = True)
            for key, (_, queries, _) in self.corpus_loaders.items()
        }
        self.similarities = {}
        for key_1, vocab_1 in vocabs.items():
            for key_2, vocab_2 in vocabs.items():
                if key_1 != key_2: 
                    if key_1 not in self.similarities:
                        self.similarities[key_1] = []
                    data_1, data_2 = vocab_tuple(key_1, vocab_1), vocab_tuple(key_2, vocab_2)
                    self.similarities[key_1].append({key_2: normalized_jaccard_similarity(data_1, data_2)})
        
        with open(output_file, 'w') as writer:
            json.dump(self.similarities, writer)
        return self.similarities
    
    
    def check_query_type_distribution(self, out_folder: str = None):
        splits = ["train" if corpus == "msmarco" else "test" for corpus in self.corpuses]
        self.load_corpuses(splits)      
    
    def check_query_answer_lexical_overlap(self, out_folder: str = None):
        self.query_overlaps = {}
        if not os.path.exists(out_folder):  
            for key, (corpus, queries, qrels) in self.corpus_loaders.items():
                query_overlap = {}
                for key,val in tqdm.tqdm(qrels.items()):
                    if key not in query_overlap:
                        query_overlap[key] = {}
                    query = queries[key]
                    for doc_id in val.keys():
                        try:
                            doc = corpus[doc_id]['text']
                            if corpus[doc_id]['title'] != '':
                                doc = corpus[doc_id]['title'] + " " + doc
                            query_overlap[key][doc_id] = calculate_overlap(query, doc)
                        except KeyError as e:
                            print(f'Got {e}, ignoring')
                if out_folder is not None:
                    if not os.path.exists(os.path.split(out_folder)[0]):
                        os.makedirs(os.path.split(out_folder)[0])
                    with open(out_folder, "w") as f:
                        json.dump(query_overlap, f)
                self.query_overlaps[key] = query_overlap

