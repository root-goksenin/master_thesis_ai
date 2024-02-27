import json 
from typing import Dict 
import os
import tqdm 

def find_scores_json(base, new=[]):
    # Find scores.json recursively, and return the list of paths.
    if os.path.exists(os.path.join(base, 'scores.json')):
        new.append(os.path.join(base, 'scores.json'))    
    for dir in os.listdir(base):
        if os.path.isdir(os.path.join(base, dir)):
            find_scores_json(os.path.join(base, dir), new)
    return new

def sort_json(file: str): 
    with open(file, 'r') as f: 
        data : Dict[str, Dict[str, float]] = json.load(f)
    sorted_dict = {
        key: dict(sorted(data[key].items(), key=lambda kv: kv[1], reverse=True))
        for key in data
    }

    base = os.path.join(os.path.split(file)[0], "sorted_scores.json")
    with open(base, 'w') as f: 
        json.dump(
                sorted_dict,
                f,
                indent=4,
            )
if __name__ == "__main__":
    scores = find_scores_json("./zero_shot_results")
    for score in tqdm.tqdm(scores):
        if not os.path.exists(os.path.join(os.path.split(score)[0], "sorted_scores.json")):
            sort_json(score)