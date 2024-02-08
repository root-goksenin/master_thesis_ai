import networkx as nx
import matplotlib.pyplot as plt
import os
import json 
# Create a sample weighted graph (you can replace this with your own graph data)
G = nx.Graph()
edges = [] 

with open(os.path.join('jaccard_similarities_beir.json'), 'r') as writer:
    similarity_matrix = json.load(writer)       


row_column_names = list(similarity_matrix.keys())
edges = []
for row in row_column_names:
    for col in row_column_names:
        if row != col:
            edge =  (row, col, {"weight" : [dict_val[col] for dict_val in similarity_matrix[row] if col in dict_val][0]})
            edges.append(edge)
    
G.add_edges_from(edges)

# Use the force-directed layout algorithm
pos = nx.spring_layout(G)

# Extract edge weights to determine edge thickness
edge_weights = [G[u][v]['weight'] for u, v in G.edges()]

# Draw the graph with edge weights
nx.draw(G, pos, with_labels=True, font_weight='bold', node_size=700, node_color='skyblue', font_color='black',
        edge_color='gray', width=edge_weights, edge_cmap=plt.cm.Blues)

# Display edge weights on the plot
edge_labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

# Show the plot
plt.savefig("network_plot.png")