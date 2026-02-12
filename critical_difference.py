import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sp

metric = "VUS-PR"
uni_metrics_baselines_path = f"TSB-AD-results/uni_mergedTable_{metric}.csv"
uni_metrics_ts2i_path = f"results/uni_mergedTable_{metric}.csv"

df_baselines = pd.read_csv(uni_metrics_baselines_path)
df_baselines["dataset"] = df_baselines["file"].str.extract(r"^\d+_([^_]+)_")[0]

df_ts2i = pd.read_csv(uni_metrics_ts2i_path)
df_ts2i["dataset"] = df_ts2i["file"].str.extract(r"^\d+_([^_]+)_")[0]

metadata_cols = [
    "file",
    "ts_len",
    "anomaly_len",
    "num_anomaly",
    "avg_anomaly_len",
    "anomaly_ratio",
    "point_anomaly",
    "seq_anomaly",
    "dataset",
]

baselines_cols = [col for col in df_baselines.columns if col not in metadata_cols]
ts2i_cols = [col for col in df_ts2i.columns if col not in metadata_cols]

agg_baselines = df_baselines.groupby("dataset")[baselines_cols].mean()
agg_ts2i = df_ts2i.groupby("dataset")[ts2i_cols].mean()

aggregated_mean = pd.concat([agg_baselines, agg_ts2i], axis=1)
aggregated_transposed = aggregated_mean.T

baseline_order = [
    "Sub-PCA",
    "KShapeAD",
    "POLY",
    "Series2Graph",
    "MOMENT (FT)",
    "MOMENT (ZS)",
    "KMeansAD",
    "USAD",
    "Sub-KNN",
    "MatrixProfile",
    "SAND",
    "CNN",
    "LSTMAD",
    "SR",
    "TimesFM",
    "IForest",
    "OmniAnomaly",
    "Lag-Llama",
    "Chronos",
    "TimesNet",
    "AutoEncoder",
    "TranAD",
    "FITS",
    "Sub-LOF",
    "OFA",
    "Sub-MCD",
    "Sub-HBOS",
    "Sub-OCSVM",
    "Sub-IForest",
    "Donut",
    "LOF",
    "AnomalyTransformer",
]

ts2i_order = [
    "CNN-AE (RN)",
    "CNN-AE (LP)",
    "CNN-AE (SG)",
    "CNN-AE (GASF)",
    "CNN-AE (GADF)",
    "CNN-AE (MTF)",
    "CNN-AE (RP)",
    "CNN-AE (RWT)",
    "CNN-AE (MWT)",
    "CNN-AE (SPIRAL)",
    "ResNet18-AE-FE (RN)",
    "ResNet18-AE-FE (LP)",
    "ResNet18-AE-FE (SG)",
    "ResNet18-AE-FE (GASF)",
    "ResNet18-AE-FE (GADF)",
    "ResNet18-AE-FE (MTF)",
    "ResNet18-AE-FE (RP)",
    "ResNet18-AE-FE (RWT)",
    "ResNet18-AE-FE (MWT)",
    "ResNet18-AE-FE (SPIRAL)",
    "ResNet18-AE-PU (RN)",
    "ResNet18-AE-PU (LP)",
    "ResNet18-AE-PU (SG)",
    "ResNet18-AE-PU (GASF)",
    "ResNet18-AE-PU (GADF)",
    "ResNet18-AE-PU (MTF)",
    "ResNet18-AE-PU (RP)",
    "ResNet18-AE-PU (RWT)",
    "ResNet18-AE-PU (MWT)",
    "ResNet18-AE-PU (SPIRAL)",
    "ResNet18-AE-DLR (RN)",
    "ResNet18-AE-DLR (LP)",
    "ResNet18-AE-DLR (SG)",
    "ResNet18-AE-DLR (GASF)",
    "ResNet18-AE-DLR (GADF)",
    "ResNet18-AE-DLR (MTF)",
    "ResNet18-AE-DLR (RP)",
    "ResNet18-AE-DLR (RWT)",
    "ResNet18-AE-DLR (MWT)",
    "ResNet18-AE-DLR (SPIRAL)",
    "PVT-v2-b1-AE-FE (RN)",
    "PVT-v2-b1-AE-FE (LP)",
    "PVT-v2-b1-AE-FE (SG)",
    "PVT-v2-b1-AE-FE (GASF)",
    "PVT-v2-b1-AE-FE (GADF)",
    "PVT-v2-b1-AE-FE (MTF)",
    "PVT-v2-b1-AE-FE (RP)",
    "PVT-v2-b1-AE-FE (RWT)",
    "PVT-v2-b1-AE-FE (MWT)",
    "PVT-v2-b1-AE-FE (SPIRAL)",
    "PVT-v2-b1-AE-PU (RN)",
    "PVT-v2-b1-AE-PU (LP)",
    "PVT-v2-b1-AE-PU (SG)",
    "PVT-v2-b1-AE-PU (GASF)",
    "PVT-v2-b1-AE-PU (GADF)",
    "PVT-v2-b1-AE-PU (MTF)",
    "PVT-v2-b1-AE-PU (RP)",
    "PVT-v2-b1-AE-PU (RWT)",
    "PVT-v2-b1-AE-PU (MWT)",
    "PVT-v2-b1-AE-PU (SPIRAL)",
    "PVT-v2-b1-AE-DLR (RN)",
    "PVT-v2-b1-AE-DLR (LP)",
    "PVT-v2-b1-AE-DLR (SG)",
    "PVT-v2-b1-AE-DLR (GASF)",
    "PVT-v2-b1-AE-DLR (GADF)",
    "PVT-v2-b1-AE-DLR (MTF)",
    "PVT-v2-b1-AE-DLR (RP)",
    "PVT-v2-b1-AE-DLR (RWT)",
    "PVT-v2-b1-AE-DLR (MWT)",
    "PVT-v2-b1-AE-DLR (SPIRAL)",
]

model_order = baseline_order + ts2i_order

dataset_order = [
    "CATSv2",
    "Daphnet",
    "Exathlon",
    "IOPS",
    "LTDB",
    "MGAB",
    "MITDB",
    "MSL",
    "NAB",
    "NEK",
    "OPPORTUNITY",
    "Power",
    "SED",
    "SMAP",
    "SMD",
    "SVDB",
    "SWaT",
    "Stock",
    "TAO",
    "TODS",
    "UCR",
    "WSD",
    "YAHOO",
]

existing_models = [m for m in model_order if m in aggregated_transposed.index]
aggregated_sorted = aggregated_transposed.reindex(existing_models)

existing_datasets = [d for d in dataset_order if d in aggregated_sorted.columns]
aggregated_sorted = aggregated_sorted[existing_datasets]

ranks = []
for dataset in existing_datasets:
    dataset_scores = aggregated_sorted[dataset].values
    rank = len(dataset_scores) + 1 - sp.rankdata(dataset_scores, method="average")
    ranks.append(rank)

ranks = np.array(ranks).T
avg_ranks = ranks.mean(axis=1)

n_datasets = len(existing_datasets)
n_models = len(existing_models)
q_alpha = 2.850  # alpha = 0.05
cd = q_alpha * np.sqrt((n_models * (n_models + 1)) / (6.0 * n_datasets))

sorted_indices = np.argsort(avg_ranks)

# top_k = 20
# sorted_indices = np.argsort(avg_ranks)[:top_k]
sorted_models = [existing_models[i] for i in sorted_indices]
sorted_ranks = avg_ranks[sorted_indices]

fig, ax = plt.subplots(figsize=(16, 26))
# fig, ax = plt.subplots(figsize=(16, 8)) <- this one was used to display top_k results

colors_cd = plt.cm.coolwarm_r(np.linspace(0.2, 0.8, len(sorted_models)))
y_positions = np.arange(len(sorted_models))
bars = ax.barh(
    y_positions,
    sorted_ranks,
    color=colors_cd,
    edgecolor="black",
    linewidth=0.5,
    alpha=0.85,
)

for i, (rank, model) in enumerate(zip(sorted_ranks, sorted_models)):
    ax.text(
        rank - 3.5,
        i - 0.05,
        f"{rank:.2f}",
        va="center",
        ha="right",
        fontsize=10,
        fontweight="bold",
        color="black",
    )

ax.set_yticks(y_positions)
ax.set_yticklabels(sorted_models, fontsize=11)
ax.set_xlabel("Average Rank (lower is better)", fontsize=14, fontweight="bold")
ax.invert_xaxis()
ax.grid(alpha=0.3, linestyle="--", linewidth=0.5, axis="x")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.axvline(
    x=sorted_ranks[0] + cd,
    color="darkred",
    linestyle="--",
    linewidth=2.5,
    label=f"CD = {cd:.2f}",
    alpha=0.8,
)
ax.legend(loc="upper right", fontsize=12, framealpha=0.9)

plt.subplots_adjust(top=0.98, bottom=0.02, left=0.18, right=0.98)
# plt.tight_layout() <- this one was used to display top_k results
ax.set_ylim(-0.5, len(sorted_models) - 0.5)
plt.savefig(f"cd_diagram_{metric.replace('-', '_')}_combined.pdf", dpi=1500)
plt.close()
