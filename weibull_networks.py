import networkx as nx
import numpy as np
import pandas as pd
from pathlib import Path

def generate_network(shape_param, n_v, n_e, seed=None):
    G = nx.empty_graph(n=n_v)

    rng = np.random.default_rng(seed=seed)

    p = rng.weibull(a=shape_param, size=n_v)

    probs = []
    for i in range(n_v):
        probs.extend(p)

    # set diagonals to zero
    for i in range(n_v):
        probs[i*n_v + i] = 0

    # normalise probs
    probs = probs / np.sum(probs, dtype=float)

    adj_mat = np.linspace(0, n_v*n_v, n_v*n_v, dtype=int, endpoint=False)
    # adj_mat = np.zeros(shape=(n_v, n_v), dtype=int)

    edge_choice = rng.choice(adj_mat, p=probs, size=n_e, replace=False)

    edge_list = []
    for x in edge_choice:
        i = int(x/n_v)
        j = x % n_v
        edge_list.append((i,j))

    G.add_edges_from(edge_list)

    return G

def simulate_activity(G, shape_param, max_requests, method="weight_edges", rng=None):
    if not rng:
        rng = np.random.default_rng()

    freqs = np.zeros(len(G.nodes())).astype(int)

    if method == "weight_edges":
        for node in G.nodes():
            this_neighbours = [x for x in nx.all_neighbors(G, node)]

            weights = rng.weibull(a=shape_param, size=len(this_neighbours))
            weights = weights / sum(weights) # normalise

            choices = rng.choice(this_neighbours, p=weights, size=max_requests, replace=True).astype(int)

            # add all choices (node IDs) to frequency list
            for c in choices:
                freqs[c] += 1

    elif method == "weight_nodes":
        node_weights = rng.weibull(a=shape_param, size=len(G.nodes()))
        node_weights = node_weights / sum(node_weights)

        node_degs = [G.degree[v] for v in G.nodes()]

        edge_weights = np.zeros(len(G.edges()))

        for j,e in enumerate(G.edges()):
            # Using convention that (u, v) is u retweeting v
            u = int(e[0])
            edge_weights[j] = node_weights[u] / node_degs[u]

        edge_weights = edge_weights / sum(edge_weights)

        # Same number of picks as method 1 i.e. N_REQUESTS per node, but this time each node wont supply an equal
        # no. requests.
        edge_choices = rng.choice(G.edges(), p=edge_weights, size=max_requests * len(G.nodes()), replace=True)

        for e in edge_choices:
            # Using convention that (u, v) is u retweeting v
            freqs[int(e[1])] += 1

    elif method == "weight_n_requests":
        node_weights = rng.weibull(a=shape_param, size=len(G.nodes()))

        requests_per_node = [min(max_requests, int(max_requests * w)) for w in node_weights]

        for node in G.nodes():
            this_neighbours = [x for x in nx.all_neighbors(G, node)]
            
            choice_indices = rng.integers(low=0, high=len(this_neighbours), size=requests_per_node[int(node)])

            for c in choice_indices:
                freqs[int(this_neighbours[c])] += 1

    df = pd.DataFrame({"freq": freqs})
    return df

def load_all_csvs(folder, pattern=None):
    if not str(folder).endswith("/"):
        folder = folder + "/"

    path_pattern = pattern
    if pattern:
        if not str(path_pattern).endswith(".csv"):
            path_pattern = folder + pattern + ".csv"
    else:
        path_pattern = folder + "*.csv"

    dataframes = []
    names = []
    # print(folder + path_pattern)
    for p in Path(".").glob(folder + path_pattern):
        filename = str(p).split(folder)[1]
        filename = filename.split(".csv")[0]
        names.append(filename)

        data = pd.read_csv(str(p))
        dataframes.append(data)

    return dataframes, names

def load_network_adjlist(folder, pattern=None):
    if not str(folder).endswith("/"):
        folder = folder + "/"

    if pattern:
        if not str(pattern).endswith(".adjlist"):
            path_pattern = folder + pattern + ".adjlist"
        else:
            path_pattern = folder + pattern
    else:
        path_pattern = folder + "*.adjlist"

    graphs = []
    names = []
    for p in Path(".").glob(folder + pattern):
        filename = str(p).split(folder)[1]
        # print(filename)
        filename = filename.split(".adjlist")[0]
        # print(filename)
        names.append(filename)

        G = nx.read_adjlist(p)
        graphs.append(G)

    return graphs, names