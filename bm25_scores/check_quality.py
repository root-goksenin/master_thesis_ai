import os 
from typing import Dict 

datasets = [
    x for x in os.listdir("../beir_data") if "cqadupstack" not in x
] + list(os.listdir("../beir_data/cqadupstack")) + list(os.listdir("../lotte_beir_format"))

exists_dict = {k : False for k in datasets}

from_path_extract_index = lambda x, index: x.split("/")[index]
from_path_extract_data_name = lambda x: from_path_extract_index(x, 1)
def find_results_json(base, new=[]):
    # Find scores.json recursively, and return the list of paths.
    if os.path.exists(os.path.join(base, 'results.json')):
        new.append(os.path.join(base, 'results.json'))    
    for dir in os.listdir(base):
        if os.path.isdir(os.path.join(base, dir)):
            find_results_json(os.path.join(base, dir), new)
    return new
def print_non_exists(dict: Dict[str, Dict[str, bool]]):
    print({k : False for k in dict if not dict[k]})
 
if __name__ == "__main__":
    results = find_results_json("./")
    for result in results:
        data = from_path_extract_data_name(result)
        exists_dict[data] = True 
    print_non_exists(exists_dict)
        