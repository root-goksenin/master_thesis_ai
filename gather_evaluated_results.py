import os
import json
import click
path = "evaluated_models"


@click.command()
@click.option("--model", type = str)
@click.option("--dataset", type = str)
def main(model, dataset):
    for dir in os.listdir(path):
        if dir == dataset:
            for sub_dir in os.listdir(os.path.join(path, dir)):
                print(sub_dir)
                if sub_dir in ["mini_lm", "three_teacher_average", "tiny_bert", "two_teacher_average"]:
                    search_dir = os.path.join(path, dir, sub_dir, "augmented_mod_no_aug")
                else:
                    search_dir = os.path.join(path, dir, sub_dir)
                for search in sorted(os.listdir(search_dir)):
                    if "False" not in search: 
                        if sub_dir != "BM25":
                            with open(os.path.join(search_dir, search, "results.json"), "r") as f:
                                data = json.load(f)
                        else:
                            with open(os.path.join(search_dir, search), "r") as f:
                                data = json.load(f)                        
                        print(round(data['ndcg']['NDCG@10'] * 100, 2))
                            
if __name__ == "__main__":
    main()                 
                    
            