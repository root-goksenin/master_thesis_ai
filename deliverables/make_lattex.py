import os 
import json 
from collections import namedtuple
import click 
output = namedtuple('Result', ['getter', 'updater'])

models = ["GPL", "msmarco-distilbert-dot-v5", "multi-qa-distilbert-dot-v1", "BM25"]
datasets = [
    x for x in os.listdir("../beir_data") if "cqadupstack" not in x
] + list(os.listdir("../beir_data/cqadupstack")) + list(os.listdir("../lotte_beir_format"))


from_path_extract_index = lambda x, index: x.split("/")[index]
from_path_extract_data_name = lambda x: from_path_extract_index(x, 2)
from_path_extract_method_name = lambda x: from_path_extract_index(x, 3)
composed = lambda x: (from_path_extract_method_name(x), from_path_extract_data_name(x))

def find_results_json(base, new=[]):
    # Find scores.json recursively, and return the list of paths.
    if os.path.exists(os.path.join(base, 'results.json')):
        new.append(os.path.join(base, 'results.json'))    
    for dir in os.listdir(base):
        if os.path.isdir(os.path.join(base, dir)):
            find_results_json(os.path.join(base, dir), new)
    return new

def return_eval_func(path, func: str): 
    with open(path, 'r') as file:
        data = json.load(file)
    for k in data:
        for k_, v_ in data[k].items():
            if k_ == func:
                return round(v_ * 100,1)

def populate_results(funcs, eval_functions):
    result_dict = {k: {k_ : {val: {} for val in eval_functions} for k_ in datasets} for k in models}
    def populate(results):
        nonlocal result_dict
        for result in results:
            method,data = funcs(result)
            for func in eval_functions:
                result_dict[method if method != "results.json" else "BM25"][data][func] = return_eval_func(result, func)
    def get_result_dict():
        return result_dict

    return output(getter=get_result_dict, updater=populate)



@click.command()
@click.argument('eval_functions', nargs = -1)
def main(eval_functions):
    eval_functions = list(eval_functions)
    zero_shot_results = find_results_json("../zero_shot_results")
    bm25_results = find_results_json("../bm25_scores")
    populator = populate_results(composed, eval_functions)
    populator.updater(zero_shot_results)
    populator.updater(bm25_results)
    latex_table = populator.getter()
    with open('table.json', 'w') as f:
        json.dump(latex_table, f, indent=4)

if __name__ == "__main__":
    main()
