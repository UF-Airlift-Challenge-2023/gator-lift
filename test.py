import numpy as np

graph = {
    0:{
        "edges":[2], 
        "weights":[2]},
    1:{
        "edges":[2, 4], 
        "weights":[2, 2]},
    2:{
        "edges":[0, 1, 3, 5], 
        "weights":[2, 2, 2, 2]},
    3:{
        "edges":[2, 6], 
        "weights":[2, 2]},
    4:{
        "edges":[1, 7], 
        "weights":[2, 1]},
    5:{
        "edges":[2, 8], 
        "weights":[2, 1]},
    6:{
        "edges":[3, 9], 
        "weights":[2, 1]},
    7:{
        "edges":[4, 8], 
        "weights":[1, 2]},
    8:{
        "edges":[5, 7, 9], 
        "weights":[1, 2, 2]},
    9:{
        "edges":[6, 8], 
        "weights":[1, 2]}
}

def convert_to_csr(graph):
    num_nodes = len(graph)
    
    offsets = []
    edges = []
    weights = []
    
    cur_offset = 0
    for node in range(num_nodes):
        offsets.append(cur_offset)
        cur_offset += len(graph[node]["edges"])
        
        edges = edges + graph[node]["edges"]
        weights = weights + graph[node]["weights"]
        
    offsets.append(cur_offset)
    
    return np.array(offsets), np.array(edges), np.array(weights)

offsets, edges, weights = convert_to_csr(graph)
print(f"offsets = {list(offsets)}")
print(f"edges =   {list(edges)}")
print(f"weights = {list(weights)}")