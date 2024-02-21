import shutil 
import glob
import os 
paths = ["pooled/dev/corpus.jsonl",
         "pooled/test/corpus.jsonl",
         "recreation/dev/corpus.jsonl",
        "recreation/test/corpus.jsonl",
        "science/dev/corpus.jsonl",
        "science/test/corpus.jsonl",
        "technology/dev/corpus.jsonl",
        "technology/test/corpus.jsonl",
        "writing/dev/corpus.jsonl",
        "writing/test/corpus.jsonl"
]


for file in glob.glob("../lotte_beir_format_new/*"):
    copy_from = "/".join([os.path.split(file)[1].split("_")[0],os.path.split(file)[1].split("_")[2], "corpus.jsonl"])
    try:
        shutil.copy(copy_from, file)
    except FileNotFoundError as e:
        print(file)
     