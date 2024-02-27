import csv
import jsonlines
import os 
import pandas as pd
import glob

def beir_corpus(corpus_file:str):
    with jsonlines.open(corpus_file) as reader:
        data = []
        try:
            for obj in reader:
                new_data = {
                    "_id": str(obj["doc_id"]),
                    "text": obj["text"],
                    "title": "",
                    "metadata": {}
                }
                data.append(new_data)
        except jsonlines.jsonlines.InvalidLineError:
            print(data[-1])
    out_file = os.path.join(os.path.split(corpus_file)[0], "corpus.jsonl")
    with jsonlines.open(out_file, mode='w') as writer:
        writer.write_all(data)
        
        
def try_(corpus_file):
    import json

    data = []
    with open(corpus_file) as f:
        for line in f:
            try:
                obj = json.loads(line)
                new_data = {
                    "_id": str(obj["doc_id"]),
                    "text": obj["text"],
                    "title": "",
                    "metadata": {}
                }
                data.append(new_data)
            except:
                print("Skipping", line)
    out_file = os.path.join(os.path.split(corpus_file)[0], "corpus.jsonl")
    with jsonlines.open(out_file, mode='w') as writer:
        writer.write_all(data)
def recursive_file_find(extension):
    return glob.glob(f"**/*collection{extension}", recursive=True)
    
if __name__ == "__main__":
    files = recursive_file_find(".jsonl")
    files.remove("writing/dev/dev_collection.jsonl")
    files.remove("writing/test/test_collection.jsonl")
    files.remove("pooled/dev/dev_collection.jsonl")
        
    for file in files:
        print(file)
        if file == "pooled/test/test_collection.jsonl":
            try_(file)