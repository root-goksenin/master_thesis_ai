import csv
import jsonlines
import os 
import pandas as pd
import glob

def beir_queries(query_file: str):
    # Load the original JSON object
    with jsonlines.open(query_file) as reader:
        data = []
        for obj in reader:
            new_data = {
                "_id": str(obj["qid"]),
                "text": obj["query"],
                "metadata": {}
            }
            data.append(new_data)
            
    out_file = os.path.join(os.path.split(query_file)[0], "queries.jsonl")

    with jsonlines.open(out_file, mode='w') as writer:
        writer.write_all(data)
        
def recursive_file_find(extension):
    return glob.glob(f"**/qas*{extension}", recursive=True)
    
if __name__ == "__main__":
    for file in recursive_file_find(".jsonl"):
        print(file)
        beir_queries(file)