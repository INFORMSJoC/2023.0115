import numpy as np
import pandas as pd
import networkx as nx
import torch
import torch_geometric as ptgeom
import time 
import dgl
from dgl.data.utils import save_graphs
import pickle as pkl
import argparse
import os 
import random


def set_random_seed(seed=7):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def load_edges_node(edgelist_path):
    with open(edgelist_path, 'r') as f:
        lines = f.readlines()
    
    edges = [list(map(int, line.split(" "))) for line in lines]
    
    edges = torch.tensor(edges, dtype=torch.int64).t()
    edges = ptgeom.utils.to_undirected(edges).t().numpy().tolist()
    
    return edges
    
def generate_node2id_dicts(edges_node):
    node2id = dict()
    for x, y in edges_node:
        if x not in node2id:
            node2id[x] = len(node2id)
        if y not in node2id:
            node2id[y] = len(node2id)
    return node2id

def get_edge_set(edges):
    """
    Args:
        edges (list): list of undirected / directed edges 

    Returns:
        set: set of directed edges
    """
    edge_set = [(e[0], e[1]) for e in edges]
    edge_set = set(edge_set)
    
    return edge_set

def get_mst_edge_set(edges):
    """
    Args:
        edges (list): list of undirected edges
        
    Returns:
        set: set of directed edges in the minimum spanning tree
    """
    g = nx.Graph()
    g.add_edges_from(edges)
    
    mstl = list(nx.tree.minimum_spanning_edges(g, algorithm="kruskal", data=False))
    mstl = [e if e[0] < e[1] else (e[1], e[0])   for e in mstl]
    
    mst_edge_set = set(mstl)
    return mst_edge_set

def generate_mst_twin_domain_dataset(edges_node, node2id, p=0.5):
    """
    Args:
        edges_node: list of undirected edges, the node are the "real" node in the dataset
        node2id: dict
        p: the proportion of the number of edges in domain1 to the number_of_edges in domain2 
    """
    id2node = dict((val, key) for key, val in node2id.items())
    
    edges = [[node2id[x], node2id[y]] for x, y in edges_node]
    
    edge_set = get_edge_set(edges)
    mst_edge_set = get_mst_edge_set(edges)
    mst_edges_node = [[id2node[e[0]], id2node[e[1]]] for e in list(mst_edge_set)]
    
    
    edge_remained_set = edge_set - mst_edge_set
    
    edge_remained_directed = list(edge_remained_set)
    edge_remained_directed = [list(item)   for item in edge_remained_directed]
    
    edge_selected_domain1_node = []
    edge_selected_domain2_node = []
    for edge in edge_remained_directed:
        if np.random.rand() < p:
            edge_selected_domain1_node.append([id2node[edge[0]], id2node[edge[1]]])

        else:
            edge_selected_domain2_node.append([id2node[edge[0]], id2node[edge[1]]])
    
    edge_selected_domain1_node = edge_selected_domain1_node + mst_edges_node
    edge_selected_domain2_node = edge_selected_domain2_node + mst_edges_node
            
    return edge_selected_domain1_node, edge_selected_domain2_node
    
def save_records(records, save_path):
    with open(save_path, 'w') as f:
        for record in records:
            content = " ".join(map(str, record)) + "\n"
            f.write(content)
    
def generate_linklabel(edges, neg2pos_ratio):
    nodes = []
    for edge in edges:
        nodes.extend(edge)
    nodes = list(set(nodes))
    
    edge_set = get_edge_set(edges)
    
    edge_num = len(edges)
    negative_edge_num = neg2pos_ratio * edge_num
    negative_edge_set = set()
    
    count = 0
    while count < negative_edge_num:
        edge_new = tuple(np.sort(np.random.choice(nodes, 2, replace=False)))
        if edge_new not in edge_set and edge_new not in negative_edge_set:
            negative_edge_set.add(edge_new)
            count += 1
    negative_edge = [[item[0], item[1]] for item in list(negative_edge_set)]
    
    ans = [[item[0], item[1], 1]   for item in edge_set] + [[item[0], item[1], 0] for item in negative_edge]
    ans = np.array(ans)
    idx = list(range(len(ans)))
    np.random.shuffle(idx)
    ans = ans[idx]
    return ans.tolist()


def generate_graph_for_bin(edges_node, node2id):
    edge_index = [[node2id[e[0]], node2id[e[1]]] for e in edges_node]
    edge_index = torch.tensor(edge_index, dtype=torch.int64).t()
    edge_index, _ = ptgeom.utils.remove_self_loops(edge_index)
    edge_index = ptgeom.utils.to_undirected(edge_index)
    
    graph = dgl.DGLGraph((edge_index[0], edge_index[1]))
    
    return graph 
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    args = parser.parse_args()
    
    set_random_seed()
    
    link_prediction_root = f'data/link_prediction/{args.dataset}'
    node_classification_root = f'data/node_classification/{args.dataset}'
    if not os.path.isdir(link_prediction_root):
        os.makedirs(link_prediction_root)
    if not os.path.isdir(node_classification_root):
        os.makedirs(node_classification_root)
    
    raw_root = f'data/raw_data/{args.dataset}'
    raw_dataset_edgelist_path = f'data/raw_data/{args.dataset}/{args.dataset}.edgelist'
    
    edges_node = load_edges_node(raw_dataset_edgelist_path) # list of undirected edges,
    
    # generate node2id
    node2id = generate_node2id_dicts(edges_node)
    if not os.path.isdir('data/node2id_dicts'):
        os.mkdirs('data/node2id_dicts')
    pkl.dump(node2id, open(f'data/node2id_dicts/{args.dataset}_node2id.pkl', 'wb'))
    
    # generate mst_twin_domain's edgelist
    edge_selected_domain1, edge_selected_domain2 = generate_mst_twin_domain_dataset(edges_node, node2id, p=0.5)

    save_records(edge_selected_domain1, os.path.join(link_prediction_root, f'{args.dataset}_mst_twin_domain1.edgelist') )
    save_records(edge_selected_domain1, os.path.join(node_classification_root, f'{args.dataset}_mst_twin_domain1.edgelist'))
    save_records(edge_selected_domain1, os.path.join(link_prediction_root, f'{args.dataset}_mst_twin_domain1_4_fold.edgelist'))
    save_records(edge_selected_domain2, os.path.join(link_prediction_root, f'{args.dataset}_mst_twin_domain2.edgelist'))
    save_records(edge_selected_domain2, os.path.join(node_classification_root, f'{args.dataset}_mst_twin_domain2.edgelist'))

    # generate mst_twin_domain's edgelabel
    linklabel_domain1 = generate_linklabel(edge_selected_domain1, 1.0)
    linklabel_domain2 = generate_linklabel(edge_selected_domain2, 1.0)
    linklabel_domain1_4_fold = generate_linklabel(edge_selected_domain1, 4.0)
    
    save_records(linklabel_domain1, os.path.join(link_prediction_root, f'{args.dataset}_mst_twin_domain1.edgelabel'))
    save_records(linklabel_domain2, os.path.join(link_prediction_root, f'{args.dataset}_mst_twin_domain2.edgelabel'))
    save_records(linklabel_domain1_4_fold, os.path.join(link_prediction_root, f'{args.dataset}_mst_twin_domain1_4_fold.edgelabel'))
            
    # generate mst_twin_domain's nodelabel   
    os.system(f"cp {raw_root}/{args.dataset}.nodelabel {node_classification_root}/{args.dataset}_mst_twin_domain2.nodelabel")
    
    
    # generate bin for pretraining
    graph = generate_graph_for_bin(edge_selected_domain1, node2id)
    dgl.data.utils.save_graphs(f'data/{args.dataset}_mst_twin_domain1.bin', [graph], {'graph_sizes': torch.tensor([graph.number_of_nodes(),], dtype=torch.int64)})