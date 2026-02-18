import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

def detect_drift(reference_df, new_df):

    drift_results = {}

    for col in reference_df.columns:
        stat, p_value = ks_2samp(reference_df[col], new_df[col])
        drift_results[col] = p_value

    return drift_results


def plot_drift(reference_df, new_df, column):

    plt.figure()
    plt.hist(reference_df[column], alpha=0.5, label="Reference")
    plt.hist(new_df[column], alpha=0.5, label="New")
    plt.legend()
    plt.title(f"Drift Check: {column}")
    plt.tight_layout()

    return plt
