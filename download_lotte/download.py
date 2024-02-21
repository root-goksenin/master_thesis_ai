
import os 
import wget

datasets = ["pooled", "recreation", "science", "technology", "writing"]
splits = ["dev", "test"]
types = ["forum", "search"]


corpus_url = "https://huggingface.co/datasets/colbertv2/lotte_passages/resolve/main/{}/{}_collection.jsonl?download=true"
queries_url = "https://huggingface.co/datasets/colbertv2/lotte_passages/resolve/main/{}/{}/questions.{}.tsv?download=true"
answers_url = "https://huggingface.co/datasets/colbertv2/lotte_passages/resolve/main/{}/{}/qas.{}.jsonl?download=true"
for data in datasets:
    for split in splits:
        wget.download(corpus_url.format(data, split), out=f"{data}/{split}")