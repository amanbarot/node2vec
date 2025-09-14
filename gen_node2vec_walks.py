import networkx as nx
import random
from alias_sampler import AliasSampler

def build_graph(edge_list, directed=False):
    G = nx.DiGraph() if directed else nx.Graph()
    G.add_edges_from(edge_list)
    return G

def preprocess_transition_probs(G, p=1.0, q=1.0):
    alias_nodes = {}
    alias_edges = {}

    for node in G.nodes():
        neighbors = sorted(G.neighbors(node))
        if neighbors:
            probs = [1 / len(neighbors)] * len(neighbors)
            alias_nodes[node] = AliasSampler(probs)

    for src in G.nodes():
        for dst in G.neighbors(src):
            dst_neighbors = sorted(G.neighbors(dst))
            probs = []
            for nbr in dst_neighbors:
                if nbr == src:
                    weight = 1 / p
                elif G.has_edge(nbr, src):
                    weight = 1
                else:
                    weight = 1 / q
                probs.append(weight)
            norm_probs = [x / sum(probs) for x in probs]
            alias_edges[(src, dst)] = AliasSampler(norm_probs)

    return alias_nodes, alias_edges

def node2vec_single_walk(
        G, 
        walk_length, 
        start_node, 
        alias_nodes, 
        alias_edges
        ):
    walk = [start_node]

    while len(walk) < walk_length:
        curr = walk[-1]
        neighbors = sorted(G.neighbors(curr))
        if not neighbors:
            break
        if len(walk) == 1:
            idx = alias_nodes[curr].sample()
            walk.append(neighbors[idx])
        else:
            prev = walk[-2]
            if (prev, curr) not in alias_edges:
                print("Issue with walks")
                break
            dst_neighbors = sorted(G.neighbors(curr))
            idx = alias_edges[(prev, curr)].sample()
            walk.append(dst_neighbors[idx])
    return walk

def simulate_walks(G, num_walks, walk_length, p=1.0, q=1.0):
    alias_nodes, alias_edges \
        = preprocess_transition_probs(G, p, q)
    nodes = list(G.nodes())
    walks = []

    for _ in range(num_walks):
        random.shuffle(nodes)
        for node in nodes:
            walk = node2vec_single_walk(
                G, walk_length, node, alias_nodes, alias_edges
                )
            walks.append(walk)

    return walks
