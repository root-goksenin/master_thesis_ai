import csv
import jsonlines
import os 
import pandas as pd
import glob



def _generate_qrels(data_frame):  # sourcery skip: avoid-builtin-shadow
    df = {"query-id": [], "corpus-id": [], "score": []}
    for _, row in data_frame.iterrows():
        for answer in row["answer_pids"]:
            id = row["qid"]
            df["query-id"].append(f"{id}")
            df["corpus-id"].append(answer)
            df["score"].append("1")
    return pd.DataFrame(df)

def beir_qrels(query_file: str):
    df = pd.read_json(query_file, lines=True)
    qrels_df = _generate_qrels(df)
    base, _ = os.path.split(query_file)
    name = "dev.tsv" if "dev" in base else "test.tsv"
    out_file = os.path.join(base, name)
    qrels_df.to_csv(out_file, sep="\t", index=False)

        
def recursive_file_find(extension):
    return glob.glob(f"**/qas*{extension}", recursive=True)
    
if __name__ == "__main__":
    for file in recursive_file_find(".jsonl"):
        print(file)
        beir_qrels(file)