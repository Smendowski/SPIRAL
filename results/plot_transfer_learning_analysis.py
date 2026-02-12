import sys
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import ast
from glob import glob
import os

METRIC = sys.argv[1] if len(sys.argv) > 1 else "VUS_PR"
METRIC_LABEL = METRIC.replace("_", "-")
print(f"Running TL strategy analysis for metric: {METRIC}")
os.makedirs(f"tl_analysis/{METRIC}", exist_ok=True)
os.makedirs(f"tl_analysis/TRAINING_TIME", exist_ok=True)

TS2I_ORDER = ["RN", "LP", "SG", "GASF", "GADF", "MTF", "RP", "RWT", "MWT", "SPIRAL"]

# Map model -> (backbone_family, tl_strategy)
MODEL_MAP = {
    "resnet18_autoencoder_frozen": ("ResNet18-AE", "FE"),
    "resnet18_autoencoder_progressive_unfreeze": ("ResNet18-AE", "PU"),
    "resnet18_autoencoder_differential_lr": ("ResNet18-AE", "DLR"),
    "pvt_autoencoder_frozen": ("PVT-v2-b1-AE", "FE"),
    "pvt_autoencoder_progressive_unfreeze": ("PVT-v2-b1-AE", "PU"),
    "pvt_autoencoder_differential_lr": ("PVT-v2-b1-AE", "DLR"),
}

TL_ORDER = ["FE", "PU", "DLR"]
TL_COLORS = {
    "FE": plt.cm.coolwarm(0.05),
    "PU": plt.cm.coolwarm(0.5),
    "DLR": plt.cm.coolwarm(0.95),
}
TL_LABELS = {
    "FE": "Frozen Encoder (FE)",
    "PU": "Progressive Unfreeze (PU)",
    "DLR": "Differential LR (DLR)",
}


plt.rcParams.update(
    {
        "font.size": 10,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def parse_curve(val):
    if isinstance(val, (list, np.ndarray)):
        return np.array(val, dtype=float)
    try:
        return np.array(ast.literal_eval(str(val)), dtype=float)
    except Exception:
        return np.array([])


def epoch_to_convergence(curve):
    if len(curve) < 2:
        return np.nan
    L0, LT = curve[0], curve[-1]
    if L0 <= LT:
        return np.nan
    target = L0 - 0.9 * (L0 - LT)
    hits = np.where(curve <= target)[0]
    return int(hits[0]) + 1 if len(hits) else len(curve)


def plateau_cv(curve):
    tail = curve[int(0.8 * len(curve)) :]
    if len(tail) < 2:
        return np.nan
    mu = np.mean(tail)
    if mu <= 0:
        return np.nan
    return np.std(tail) / mu


def style_violin(parts, colors):
    for pc, col in zip(parts["bodies"], colors):
        pc.set_facecolor(col)
        pc.set_alpha(0.78)
        pc.set_edgecolor("white")
        pc.set_linewidth(0.5)
    parts["cmedians"].set_color("black")
    parts["cmedians"].set_linewidth(2)
    for key in ["cmins", "cmaxes", "cbars"]:
        parts[key].set_color("#888888")
        parts[key].set_linewidth(0.9)


files = glob("*.csv")
files = [f for f in files if not f.startswith(("uni_mergedTable", "00_BACKFILL"))]
print(f"Loading {len(files)} files...")
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

for col in [
    "AUC_PR",
    "AUC_ROC",
    "VUS_PR",
    "VUS_ROC",
    "Standard_F1",
    "PA_F1",
    "Event_F1",
    "R_F1",
    "Affiliation_F",
    "training_time_s",
]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

assert METRIC in df.columns, f"Metric '{METRIC}' not found!"

df["train_curve"] = df["train_losses"].apply(parse_curve)
df["n_epochs"] = df["train_curve"].apply(len)
df["epoch_conv"] = df["train_curve"].apply(epoch_to_convergence)
df["plateau_cv"] = df["train_curve"].apply(plateau_cv)
df["ts2i"] = df["ts2i_transformation_pretty_name"].str.strip().str.upper()
df["dataset"] = df["file_path"].str.extract(r"^\d+_([^_]+)_")[0]

# Assign backbone and TL strategy. Drop CNN-AE as this bacbone has no TL strategy
df["backbone"] = df["model"].map({k: v[0] for k, v in MODEL_MAP.items()})
df["tl_strategy"] = df["model"].map({k: v[1] for k, v in MODEL_MAP.items()})
df = df[df["backbone"].notna() & (df["n_epochs"] > 5)].copy()

tl_present = [s for s in TL_ORDER if s in df["tl_strategy"].unique()]
colors_tl = [TL_COLORS[s] for s in tl_present]

per_ds_tl = (
    df.groupby(["tl_strategy", "dataset"])
    .agg(
        metric=(METRIC, "mean"),
        epoch_conv=("epoch_conv", "mean"),
        stability=("plateau_cv", "mean"),
        train_time=("training_time_s", "mean"),
        n_epochs=("n_epochs", "mean"),
    )
    .reset_index()
)

agg_tl = (
    per_ds_tl.groupby("tl_strategy")
    .agg(
        metric_mean=("metric", "mean"),
        metric_std=("metric", "std"),
        metric_n=("metric", "count"),
        epoch_mean=("epoch_conv", "mean"),
        epoch_std=("epoch_conv", "std"),
        stab_mean=("stability", "mean"),
        stab_std=("stability", "std"),
        time_mean=("train_time", "mean"),
        time_std=("train_time", "std"),
    )
    .reindex(tl_present)
)
agg_tl["metric_sem"] = agg_tl["metric_std"] / np.sqrt(agg_tl["metric_n"])
agg_tl["epoch_sem"] = agg_tl["epoch_std"] / np.sqrt(agg_tl["metric_n"])
agg_tl["time_sem"] = agg_tl["time_std"] / np.sqrt(agg_tl["metric_n"])

print(f"\n{'=' * 70}")
print(f"TL STRATEGY STATISTICS — {METRIC_LABEL}")
print(f"{'=' * 70}")
print(
    agg_tl[
        [
            "metric_mean",
            "metric_sem",
            "epoch_mean",
            "epoch_sem",
            "stab_mean",
            "stab_std",
            "time_mean",
            "time_sem",
        ]
    ]
    .round(4)
    .to_string()
)

print(f"\nRELATIVE TO FE:")
fe = agg_tl.loc["FE"]
for s in tl_present:
    if s == "FE":
        continue
    r = agg_tl.loc[s]
    print(
        f"  {s}: {METRIC_LABEL} {r['metric_mean'] - fe['metric_mean']:+.4f} | "
        f"time {r['time_mean'] / fe['time_mean']:.2f}x | "
        f"epochs {100 * (r['epoch_mean'] - fe['epoch_mean']) / (fe['epoch_mean'] + 1e-9):+.1f}%"
    )

gap_tl = agg_tl["metric_mean"].max() - agg_tl["metric_mean"].min()
print(f"\nGap TL strategies: {gap_tl:.4f}")

fig, ax = plt.subplots(figsize=(7, 5.5))

for s in tl_present:
    r = agg_tl.loc[s]
    col = TL_COLORS[s]
    size = np.clip(r["stab_mean"] * 1200, 60, 700)

    ax.plot(
        [r["epoch_mean"] - r["epoch_sem"], r["epoch_mean"] + r["epoch_sem"]],
        [r["metric_mean"], r["metric_mean"]],
        color=col,
        lw=1.5,
        alpha=0.7,
        zorder=3,
    )
    ax.plot(
        [r["epoch_mean"], r["epoch_mean"]],
        [r["metric_mean"] - r["metric_sem"], r["metric_mean"] + r["metric_sem"]],
        color=col,
        lw=1.5,
        alpha=0.7,
        zorder=3,
    )
    ax.scatter(
        r["epoch_mean"],
        r["metric_mean"],
        s=size,
        color=col,
        edgecolors="white",
        lw=1.0,
        alpha=0.92,
        zorder=4,
    )
    ax.annotate(
        TL_LABELS[s],
        (r["epoch_mean"], r["metric_mean"]),
        textcoords="offset points",
        xytext=(8, 4),
        fontsize=9,
        color=col,
        fontweight="bold",
    )

ax.axvline(agg_tl["epoch_mean"].mean(), color="grey", lw=0.7, ls=":", alpha=0.5)
ax.axhline(agg_tl["metric_mean"].mean(), color="grey", lw=0.7, ls=":", alpha=0.5)
ax.set_xlabel("Mean epochs to reach 90% training loss reduction")
ax.set_ylabel(f"Mean {METRIC_LABEL}")
ax.grid(alpha=0.12, ls="--")
plt.tight_layout()
plt.savefig(
    f"tl_analysis/{METRIC}/bubble_tl_{METRIC}.pdf", bbox_inches="tight", dpi=1500
)
plt.close()

per_ds_ts2i_tl_metric = (
    df.groupby(["ts2i", "tl_strategy", "dataset"])
    .agg(metric=(METRIC, "mean"))
    .reset_index()
)
per_ds_ts2i_tl_time = (
    df.groupby(["ts2i", "tl_strategy", "dataset"])
    .agg(train_time=("training_time_s", "mean"))
    .reset_index()
)

for t in TS2I_ORDER:
    sub = per_ds_ts2i_tl_metric[per_ds_ts2i_tl_metric["ts2i"] == t]
    if sub.empty:
        continue
    data = [sub[sub["tl_strategy"] == s]["metric"].dropna().values for s in tl_present]

    fig, ax = plt.subplots(figsize=(5, 4))
    parts = ax.violinplot(
        data,
        positions=np.arange(len(tl_present)),
        showmedians=True,
        showextrema=True,
        widths=0.55,
    )
    style_violin(parts, colors_tl)

    for i, (s, vals) in enumerate(zip(tl_present, data)):
        if len(vals) == 0:
            continue
        ax.scatter(
            i,
            np.mean(vals),
            marker="D",
            s=40,
            color=TL_COLORS[s],
            edgecolors="white",
            lw=0.8,
            zorder=5,
        )
        offset = 0.03 * (max(vals) - min(vals)) if max(vals) != min(vals) else 0.01
        ax.text(
            i,
            np.mean(vals) + offset * 1.5,
            f"{np.mean(vals):.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_xticks(np.arange(len(tl_present)))
    ax.set_xticklabels(tl_present)
    ax.set_ylabel(f"Mean {METRIC_LABEL}")
    ax.grid(alpha=0.12, ls="--", axis="y")
    plt.tight_layout()
    plt.savefig(
        f"tl_analysis/{METRIC}/tl_strategy_{t}_{METRIC}.pdf",
        bbox_inches="tight",
        dpi=1500,
    )
    plt.close()

    sub_t = per_ds_ts2i_tl_time[per_ds_ts2i_tl_time["ts2i"] == t]
    data_t = [
        sub_t[sub_t["tl_strategy"] == s]["train_time"].dropna().values
        for s in tl_present
    ]

    fig, ax = plt.subplots(figsize=(5, 4))
    parts = ax.violinplot(
        data_t,
        positions=np.arange(len(tl_present)),
        showmedians=True,
        showextrema=True,
        widths=0.55,
    )
    style_violin(parts, colors_tl)

    for i, (s, vals) in enumerate(zip(tl_present, data_t)):
        if len(vals) == 0:
            continue
        ax.scatter(
            i,
            np.mean(vals),
            marker="D",
            s=40,
            color=TL_COLORS[s],
            edgecolors="white",
            lw=0.8,
            zorder=5,
        )
        offset = 0.03 * (max(vals) - min(vals)) if max(vals) != min(vals) else 0.5
        ax.text(
            i,
            np.mean(vals) + offset * 1.5,
            f"{np.mean(vals):.1f}s",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_xticks(np.arange(len(tl_present)))
    ax.set_xticklabels(tl_present)
    ax.set_ylabel("Mean training time (s)")
    ax.grid(alpha=0.12, ls="--", axis="y")
    plt.tight_layout()
    plt.savefig(
        f"tl_analysis/TRAINING_TIME/tl_strategy_{t}_training_time_s.pdf",
        bbox_inches="tight",
        dpi=1500,
    )
    plt.close()

print(f"\n{'=' * 70}")
print(f"PER TS2I × TL STRATEGY MEANS — {METRIC_LABEL}")
print(f"{'=' * 70}")
for t in TS2I_ORDER:
    sub = per_ds_ts2i_tl_metric[per_ds_ts2i_tl_metric["ts2i"] == t]
    if sub.empty:
        continue
    marker = " <- SPIRAL" if t == "SPIRAL" else ""
    print(f"\n  {t}{marker}")
    for s in tl_present:
        vals = sub[sub["tl_strategy"] == s]["metric"].dropna().values
        if len(vals) == 0:
            continue
        print(
            f"    {TL_LABELS[s]:35s}:  mean={np.mean(vals):.4f}  "
            f"median={np.median(vals):.4f}  std={np.std(vals):.4f}  n={len(vals)}"
        )

print(f"\n{'=' * 70}")
print("PER TS2I × TL STRATEGY MEANS — TRAINING TIME (s)")
print(f"{'=' * 70}")
for t in TS2I_ORDER:
    sub = per_ds_ts2i_tl_time[per_ds_ts2i_tl_time["ts2i"] == t]
    if sub.empty:
        continue
    marker = " <- SPIRAL" if t == "SPIRAL" else ""
    print(f"\n  {t}{marker}")
    for s in tl_present:
        vals = sub[sub["tl_strategy"] == s]["train_time"].dropna().values
        if len(vals) == 0:
            continue
        print(
            f"    {TL_LABELS[s]:35s}:  mean={np.mean(vals):.2f}s  "
            f"median={np.median(vals):.2f}s  std={np.std(vals):.2f}s  n={len(vals)}"
        )
