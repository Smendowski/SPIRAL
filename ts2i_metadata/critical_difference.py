import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sp

metric = "TS2I-Throughput"
df = pd.read_csv("ts2i_metadata_combined.csv")

df["dataset"] = df["file_path"].str.extract(r"^\d+_([^_]+)_")[0]

model_cols = df["ts2i_transformation_pretty_name"].unique().tolist()
model_order = model_cols
dataset_order = df["dataset"].unique().tolist()

aggregated_mean = (
    df.groupby(["dataset", "ts2i_transformation_pretty_name"])[
        "test_ts2i_transformation_throughput_images_per_sec"
    ]
    .mean()
    .unstack()
)

aggregated_transposed = aggregated_mean.T
aggregated_sorted = aggregated_transposed.reindex(model_order).astype(float)
existing_datasets = [d for d in dataset_order if d in aggregated_sorted.columns]
aggregated_sorted = aggregated_sorted[existing_datasets]

ranks = []
for dataset in existing_datasets:
    dataset_scores = aggregated_sorted[dataset].values
    rank = len(dataset_scores) + 1 - sp.rankdata(dataset_scores, method="average")
    ranks.append(rank)

ranks = np.array(ranks).T
avg_ranks = ranks.mean(axis=1)

# Critical Difference
n_datasets = len(existing_datasets)
n_models = len(model_order)
q_alpha = 2.850
cd = q_alpha * np.sqrt((n_models * (n_models + 1)) / (6.0 * n_datasets))

# Sort
sorted_indices = np.argsort(avg_ranks)
sorted_models = [model_order[i] for i in sorted_indices]
sorted_ranks = avg_ranks[sorted_indices]


fig, ax = plt.subplots(figsize=(7, 3.5))
colors_cd = plt.cm.coolwarm_r(np.linspace(0.2, 0.8, len(sorted_models)))

y_positions = np.arange(len(sorted_models))
ax.barh(
    y_positions,
    sorted_ranks,
    height=0.6,
    color=colors_cd,
    edgecolor="black",
    linewidth=0.3,
    alpha=0.85,
)

for i, (rank, model) in enumerate(zip(sorted_ranks, sorted_models)):
    ax.text(
        rank - 0.45,
        i - 0.05,
        f"{rank:.2f}",
        va="center",
        ha="right",
        fontsize=8,
        fontweight="bold",
        color="black",
    )

ax.set_yticks(y_positions)
ax.set_yticklabels(sorted_models, fontsize=9)
ax.set_xlabel("Average Rank (lower is better)", fontsize=10, fontweight="bold")

ax.invert_xaxis()
ax.grid(alpha=0.25, linestyle="--", linewidth=0.4, axis="x")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.axvline(
    x=sorted_ranks[0] + cd,
    color="darkred",
    linestyle="--",
    linewidth=1.2,
    label=f"CD = {cd:.2f}",
    alpha=0.7,
)

ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

plt.subplots_adjust(top=0.98, bottom=0.12, left=0.08, right=0.98)
ax.set_ylim(-0.5, len(sorted_models) - 0.5)

plt.savefig("cd_diagram_ts2i_throughput.pdf", dpi=300)
plt.close()

print(f"\nCritical Difference: {cd:.2f}")
