import csv
import jsonlines
import os 
import pandas as pd
import glob
def _read_tsv_file(file_path):
    data_ = []
    with open(file_path, 'r', newline='', encoding='utf-8') as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')
        for row in reader:
            data = {"_id": str(row[0]), "title": "", "text": str(row[1]), "metadata": {}}
            data_.append(data)
    return data_

def tsv_corpus_to_beir(corpus_file: str):
    data = _read_tsv_file(corpus_file)
    out_file = os.path.join(os.path.split(corpus_file)[0], "corpus.jsonl")
    with jsonlines.open(out_file, mode='w') as writer:
        writer.write_all(data)
        
def recursive_file_find(extension):
    return glob.glob(f"**/collection{extension}", recursive=True)
    
if __name__ == "__main__":
    for file in recursive_file_find(".tsv"):
        print(file)
        if file == "pooled/test/collection.tsv":
            tsv_corpus_to_beir(file)