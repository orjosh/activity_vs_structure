import weibull_networks as wb
import numpy as np
import pandas as pd
import networkx as nx

### Generating the networks ###################################################
N_NODES = int(1e4) # Each network has this many nodes and this many edges
N_EDGES = int(1e6)

N_REPS = 1 # How many networks to generate per shape parameter

SEEDS = np.linspace(0, N_REPS, N_REPS).astype(int) # Each rep. has a different RNG seed, for reproducibility

NETWORKS_SAVE_PATH = "networks/" # This folder must exist prior to running
NAME_FORMAT = "weibull_var{var:.2f}_{point_i}-{rep_j}.adjlist"

df = pd.read_csv("weibull_shape_variance.csv") # pre-calculated with `get_logspace_weibull_variances.py`
shape_params = df["shape"].to_list()
variances = df["variance"].to_list()

for i,a in enumerate(shape_params):
    for j in range(N_REPS):
        print(f"Generating network {i+1}/{len(shape_params)} (shape={a:.3f}), repetition {j+1}/{N_REPS}")
        G = wb.generate_network(shape_param=a, n_v=N_NODES, n_e=N_EDGES, seed=SEEDS[j])

        nx.write_adjlist(G, NETWORKS_SAVE_PATH+NAME_FORMAT.format(var=variances[i], point_i=i, rep_j=j))

### Simulating activity #######################################################
N_REQUESTS = 3000 # Number of times each node supplies a retweet (depends on method)
FREQS_CSV_SAVE_PATH = "freqs/"

repeat_indices = np.linspace(0, N_REPS, N_REPS).astype(int)

for r in repeat_indices:
    rng = np.random.default_rng(r)

    for i in range(len(shape_params)):
        pattern = f"*_{i}-{r}*.adjlist"
        graphs, filenames = wb.load_network_adjlist(NETWORKS_SAVE_PATH, pattern=pattern) # will only load 1 graph

        G = graphs[0]

        for k,shape in enumerate(shape_params):
            print(f"Simulating activity: network {i+1}/{len(shape_params)}, activity {k+1}/{len(shape_params)}, rep. {r}")

            # Method 1: "weight_edges"
            # Method 2: "weight_nodes"
            # Method 3: "weight_n_requests"

            df = wb.simulate_activity(G, shape, N_REQUESTS, method="weight_nodes", rng=rng)

            # i = network index, r = repetition, k = activity index
            # Assuming 1 repeat (r=0), the bottom-left cell of the phase diagram will be `freqs_nw-0-0_ac-0`, the top-left cell will be `freqs_nw-0-0_ac-15`, the top-right will be `freqs_nw_15-0_ac-15`, and the bottom-right will be `freqs_nw_15-0_ac-0`.
            df.to_csv(FREQS_CSV_SAVE_PATH + f"freqs_nw-{i}-{r}_ac-{k}.csv")