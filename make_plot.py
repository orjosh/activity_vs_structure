import numpy as np
import pandas as pd
import weibull_networks as wb
from matplotlib import pyplot as plt

N_REPS = 1
FREQS_CSV_SAVE_PATH = "freqs/"

var_shape = pd.read_csv("weibull_shape_variance.csv")
variances = var_shape["ariance"].to_list()

freq_variances = np.zeros((len(variances), len(variances)))

for i in range(len(variances)):
    for j in range(len(variances)):
        # Load all repeats for this single point (i, j)
        freq_datasets, filenames = wb.load_all_csvs(folder=FREQS_CSV_SAVE_PATH, pattern=f"*-{i}-*_ac-{j}.csv")

        # Convert to lists
        variance = 0
        for d in freq_datasets:
            f = d["freq"].to_list()
            variance += np.var(f)

        # Average over no. repetitions
        variance = variance / N_REPS

        freq_variances[j][i] = variance

fig, ax = plt.subplots()

im = ax.pcolormesh(variances, variances, freq_variances, norm='log')

fig.colorbar(im, ax=ax, label="Retweet count variance", location="bottom")

ax.set_xlabel(r"$\mathcal{W}(a)$ variance")
ax.set_ylabel(r"$\mathcal{W}(a)$ variance")
ax.set_xscale('log')
ax.set_yscale('log')

ax.set_xlim([0.07, 120000])
ax.set_ylim([0.07, 120000])

ax_x2 = ax.secondary_xaxis('top')
ax_x2.set_xscale('linear')
ax_x2.set_xlabel("Degree distribution variance")

ax_y2 = ax.secondary_yaxis('right')
ax_y2.set_yscale('linear')
ax_y2.set_ylabel("Retweet contribution distribution variance")

tick_pos = [1e3, 60000, 120000]

ax_x2.set_xticks(tick_pos, labels=[r"$3.29\times10^2$", r"$7.67\times10^3$", r"$9.28\times10^5$"])
ax_y2.set_yticks(tick_pos, labels=[r"$4.09\times10^{5}$", r"$1.67\times10^{6}$", r"$1.86\times10^{6}$"])

fig.tight_layout()
fig.set_size_inches(7,7)

fig.savefig("phasediagram_method3.png")