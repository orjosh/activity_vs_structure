import numpy as np
import pandas as pd
import scipy.stats
from matplotlib import pyplot as plt

SHAPE_PARAMS = np.linspace(0.20, 4, 3000) # Explore this range of Weibull shape params
N_SAMPLES = 100000 # Sample size for Weibull distribution

variance = []
skew = []
coeff_var = []

for i,a in enumerate(SHAPE_PARAMS):
    distr = np.random.weibull(a, N_SAMPLES)
    # variance.append(1-np.exp(-1*np.var(distr)))
    variance.append(np.var(distr))
    skew.append(scipy.stats.skew(distr))
    coeff_var.append(np.std(distr)/np.mean(distr))
    print(f"Shape param {a:.3f} ({i+1}/{len(SHAPE_PARAMS)})", end="\r")

# Take N linearly spaced samples of the y-axis (Variance), and find what the corresponding shape parameters are
N_VARIANCE_SAMPLES = 15

low_magnitude = np.log10(min(variance))
high_magnitude = np.log10(max(variance))

desired_variances = np.logspace(low_magnitude, high_magnitude, num=N_VARIANCE_SAMPLES)
found_variances = []
found_shape_params = []

for v_d in desired_variances:
    d = np.inf
    index = 0
    found_v = 0

    for i,v in enumerate(variance):
        if abs(v_d - v) < d:
            d = abs(v_d - v)
            index = i
            found_v = v
    
    found_shape_params.append(SHAPE_PARAMS[index])
    found_variances.append(found_v)

df = pd.DataFrame({'variance': found_variances, 'shape': found_shape_params})
df.to_csv('weibull_shape_variance.csv', index=False)

# # Plot variance and skewness vs. shape param
# fig, ax = plt.subplots()

# ax.scatter(SHAPE_PARAMS, variance)
# ax.set_xlabel("Shape parameter")
# ax.set_ylabel("Variance")
# ax.set_yscale('log')
# ax.set_title("Variance vs. shape parameter")
# fig.savefig("figures/weibull_variance.png")

# fig, ax = plt.subplots()

# ax.scatter(SHAPE_PARAMS, skew)
# ax.set_xlabel("Shape parameter")
# ax.set_ylabel("Skewness")
# ax.set_yscale('log')
# ax.set_title("Skewness vs. shape parameter")
# fig.savefig("figures/weibull_skewness.png")