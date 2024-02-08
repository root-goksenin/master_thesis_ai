import csv
import jsonlines
import os 
import pandas as pd
import click 

def _read_tsv_file(file_path):
    data_ = []
    with open(file_path, 'r', newline='', encoding='utf-8') as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')
        for row in reader:
            data = {"_id": str(row[0]), "title": "", "text": str(row[1]), "metadata": {}}
            data_.append(data)
    return data_

def _generate_qrels(data_frame):  # sourcery skip: avoid-builtin-shadow
    df = {"query-id": [], "corpus-id": [], "score": []}
    for _, row in data_frame.iterrows():
        for answer in row["answer_pids"]:
            id = row["qid"]
            df["query-id"].append(f"{id}")
            df["corpus-id"].append(answer)
            df["score"].append("1")
    return pd.DataFrame(df)

def corpus_to_beir(data_path, out_path):
    data = _read_tsv_file(data_path)
    with jsonlines.open(os.path.join(out_path, 'corpus.jsonl'), 'w') as writer:
        writer.write_all(data)

def queries_to_beir(data_path, out_path):
    data_style = os.path.normpath(data_path).split(os.path.sep)[1]
    
    df = pd.read_json(data_path, lines=True)
    qrels_df = _generate_qrels(df)

    qrels_path_base = os.path.join(out_path, 'qrels')
    if not os.path.exists(qrels_path_base):
        os.makedirs(qrels_path_base)
    qrels_path = os.path.join(qrels_path_base, f"{data_style}.tsv")
    qrels_df.to_csv(qrels_path, sep="\t", index=False)

    try:
        df.drop(columns=["url", "answer_pids"], inplace=True) 
    except KeyError as e:
        print(f" Got: {e}, ignoring")
        
    df.rename(columns={"qid": "_id", "query": "text"}, inplace=True)
    df['_id'] = df['_id'].astype(str)
    df['metadata'] = [{} for _ in range(len(df))]

    with open(os.path.join(out_path, 'queries.jsonl'), 'w') as f:
        print(df.to_json(orient='records', lines=True), file=f, flush=False)


@click.command()
@click.option("--data_name", type = click.Choice(["lifestyle", "pooled", "recreation", "science", "technology", "writing"]))
@click.option("--split", type = click.Choice(["dev", "test"]))
@click.option("--task", type = click.Choice(["search", "forum"]))
def lotte_to_beir(data_name, split, task):
    out_path = os.path.join("./beir_format", data_name, task, split)
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    corpus_to_beir(f"{data_name}/{split}/collection.tsv", out_path)
    queries_to_beir(f"{data_name}/{split}/qas.{task}.jsonl",out_path)   
if __name__ == "__main__":
    lotte_to_beir()
