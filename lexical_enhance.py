# Understand how the mixup effects the relevant / non-relevant queries

# 1. Plot the distribution of results with the BM25, Dense Retriver, and mix-up.
# User can select the mix-up parameter and observe the change in the score.
# Relevant documents for the query is highlted with green irrelevant ones are highlighed with red.
# Turn on and off BM25, Dense, scores. If both turned on show the mix-up dashboard.
# Plot bm25 distribution over documents for one query. Plot dense distribution over documents for one query. We need 2 histograms for each query
# Most similar, show 10 document line in the hist plot. 
# Show the query, and hovered doc on the side!
# Only show 100 datapoints.

#%%
from gpl_improved.utils import reweight_results, load_pretrained_bi_retriver
from gpl_improved.trainer.RetriverWriter import EvaluateGPL, BM25Wrapper
from beir.datasets.data_loader import GenericDataLoader
import matplotlib.pyplot as plt 
import json 
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html
from dash.dependencies import Input, Output

def generate_colormap():
    from  matplotlib.colors import LinearSegmentedColormap
    c = ["darkred","red","lightcoral","white", "palegreen","green","darkgreen"]
    v = [0,.15,.4,.5,0.6,.9,1.]
    l = list(zip(v[::-1],c))
    return LinearSegmentedColormap.from_list('rg',l, N=256)


def evaluator_factory(bi_retriver, queries = None, corpus = None, data_name = None, k_val = None):
    retriver = EvaluateGPL(bi_retriver,queries, corpus, k_val = k_val)
    if data_name is not None:
        retriver = BM25Wrapper(retriver, corpus_name=data_name, eval_bm25 = True)
    return retriver


def bi_retriver_factory(data_name, aug_strategy, model_name ):
    return load_pretrained_bi_retriver(
            data_name=data_name,
            model_name=model_name,
            aug_strategy=aug_strategy,
        )

    
class ScoreGetter:
    def __init__(self, data_name, n):
        corpus, queries, qrels = GenericDataLoader(f"./gpl_given_data/{data_name}").load(split="dev")
        self.query_relevant_docs = qrels
        retriver =  bi_retriver_factory(data_name, aug_strategy = "no_aug", model_name = "mini_lm")
        
        if os.path.exists("dense_score.json"):
            with open("dense_score.json", "r") as file:
                 self.query_dense_scores = json.load(file)
            
        else:
            self.query_dense_scores = evaluator_factory(retriver, queries, corpus, k_val = n).results()
            with open("dense_score.json", "w") as file:
                json.dump(self.query_dense_scores, file)
        
        self.query_bm25_scores = evaluator_factory(retriver,queries, corpus, data_name = data_name).results()     
        self.n = n
    
    def get_relevant_docs(self,query_id):
        return self.query_relevant_docs[query_id]


    def get_dense_doc_scores(self,query_id):
        return dict(list(self.query_dense_scores[query_id].items())[:self.n])
    

    def get_bm25_doc_scores(self,query_id):
        return dict(list(self.query_bm25_scores[query_id].items())[:self.n])
    



def plot_scores(scores,query, weight):
    fig = make_subplots(rows=3, cols=1)
    add_histogram(scores.get_bm25_doc_scores(query), scores.get_relevant_docs(query), fig, row = 1, col = 1)
    add_histogram(scores.get_dense_doc_scores(query), scores.get_relevant_docs(query), fig, row = 2, col = 1)
    mixed = mix_up(scores.get_dense_doc_scores(query), scores.get_bm25_doc_scores(query), weight = weight)
    add_histogram(mixed, scores.get_relevant_docs(query), fig, row = 3, col = 1)
    fig.update_layout(
        title='Histogram of Document Scores',
        xaxis_title='Doc Scores',
        yaxis_title='Count',
        width=800,
        height=1200,
    )
    return fig

def add_histogram(doc_scores, relevant_docs, fig, row, col):
    cmap = lambda key: 'green' if False else 'red'
    # Doc scores can have gradient of weights.
    keys = sorted(doc_scores.keys(), key=lambda item: item[1], reverse = True)[:100]
    values = sorted(list(doc_scores.values()), reverse=True)[:100]
    fig.add_trace(go.Histogram(
        x=values,
        nbinsx=len(set(values)),
        opacity=0.7,
        autobinx = False,
        histnorm='percent',
        marker=dict(color=[cmap(key) for key in keys]),
    ), row = row, col = col)

    # Add vertical line
    fig.add_shape(
        dict(
            type='line',
            x0=values[9],
            x1=values[9],
            y0=0,
            y1=10,
            line=dict(color='black'),
            label=dict(text=f"Top 10 ends here")
        ),
        row = row, col = col
    )
    
    for id, (key, value) in enumerate(sorted(doc_scores.items(), key=lambda item: item[1], reverse = True)):
        # Add vertical line
        if key in relevant_docs:
            fig.add_shape(
                dict(
                    type='line',
                    x0=value,
                    x1=value,
                    y0=0,
                    y1=10,
                    line=dict(color='green'),
                    label=dict(text=f"Rank {id}")
                ),
                row = row, col = col
            )


        
def mix_up(dense_score, bm25_score, weight):
    new_dict = {} 
    for key in dense_score.keys():
        if key in bm25_score:
            new_dict[key] = dense_score[key] + (weight * bm25_score[key]) 
        else:
            new_dict[key] = dense_score[key]
    return new_dict    


data_name = "fiqa"
scores = ScoreGetter(data_name=data_name, n = 1000)
if __name__ == "__main__":

    fig = plot_scores(scores, query = "1", weight = 3)
    app = dash.Dash(__name__)
    # Define the layout of the app
    app.layout = html.Div([
        dcc.Graph(id='histogram-plot', figure=fig),
        dcc.Dropdown(
            id='query-dropdown',
            options=[{'label': query_num, 'value': query_num} for query_num in scores.query_relevant_docs.keys()],
            value='1',  # Default value is "1"
            style={'width': '50%'},
        ),
        dcc.Slider(
            id='weight-slider',
            min=0,
            max=5,
            step=0.1,
            marks={i/10: str(i/10) for i in range(0, 51)},
            value=1.0,  # Default value is 1.0,
        )
        
    ])


    # Define the callback to update the histograms based on the selected query number and weight
    @app.callback(
        Output('histogram-plot', 'figure'),
        [Input('query-dropdown', 'value'),
            Input('weight-slider', 'value')]
    )
    def update_histogram(selected_query_num, selected_weight):
        return plot_scores(
            scores, query=selected_query_num, weight=float(selected_weight)
        )

    app.run_server(port=8001)


    # %%
    
    
    
    
