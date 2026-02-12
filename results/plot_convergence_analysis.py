import sys
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import ast
from glob import glob
import os

# AUC_PR,AUC_ROC,VUS_PR,VUS_ROC,Standard_F1,PA_F1,Event_F1,R_F1,Affiliation_F
METRIC = sys.argv[1] if len(sys.argv) > 1 else "VUS_PR"
METRIC_LABEL = METRIC.replace("_", "-")
print(f"Running analysis for metric: {METRIC}")
os.makedirs(f"convergence/{METRIC}", exist_ok=True)
os.makedirs(f"convergence/TRAINING_TIME", exist_ok=True)

TS2I_ORDER = ["RN", "LP", "SG", "GASF", "GADF", "MTF", "RP", "RWT", "MWT", "SPIRAL"]

MODEL_MAP = {
    "cnn_autoencoder": "CNN-AE",
    "resnet18_autoencoder_frozen": "ResNet18-AE",
    "resnet18_autoencoder_progressive_unfreeze": "ResNet18-AE",
    "resnet18_autoencoder_differential_lr": "ResNet18-AE",
    "pvt_autoencoder_frozen": "PVT-v2-b1-AE",
    "pvt_autoencoder_progressive_unfreeze": "PVT-v2-b1-AE",
    "pvt_autoencoder_differential_lr": "PVT-v2-b1-AE",
}

MODEL_ORDER = ["CNN-AE", "ResNet18-AE", "PVT-v2-b1-AE"]
MODEL_COLORS = {
    "CNN-AE": plt.cm.coolwarm(0.05),
    "ResNet18-AE": plt.cm.coolwarm(0.5),
    "PVT-v2-b1-AE": plt.cm.coolwarm(0.95),
}

SPIRAL_COLOR = plt.cm.coolwarm(0.95)
SPIRAL_DARK = plt.cm.coolwarm(0.90)
OTHER_COLOR = "#bbbbbb"

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
df["model_family"] = df["model"].map(MODEL_MAP)
df["ts2i"] = df["ts2i_transformation_pretty_name"].str.strip().str.upper()
df["dataset"] = df["file_path"].str.extract(r"^\d+_([^_]+)_")[0]
df = df[df["model_family"].notna() & (df["n_epochs"] > 5)].copy()


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


df["epoch_conv"] = df["train_curve"].apply(epoch_to_convergence)
df["plateau_cv"] = df["train_curve"].apply(plateau_cv)


per_dataset_ts2i = (
    df.groupby(["ts2i", "dataset"])
    .agg(
        metric=(METRIC, "mean"),
        epoch_conv=("epoch_conv", "mean"),
        stability=("plateau_cv", "mean"),
        train_time=("training_time_s", "mean"),
        n_epochs=("n_epochs", "mean"),
    )
    .reset_index()
)

agg = (
    per_dataset_ts2i.groupby("ts2i")
    .agg(
        metric_mean=("metric", "mean"),
        metric_std=("metric", "std"),
        metric_n=("metric", "count"),
        epoch_mean=("epoch_conv", "mean"),
        epoch_std=("epoch_conv", "std"),
        epoch_n=("epoch_conv", "count"),
        stab_mean=("stability", "mean"),
        stab_std=("stability", "std"),
        time_mean=("train_time", "mean"),
        n_epochs_mean=("n_epochs", "mean"),
    )
    .reindex(TS2I_ORDER)
)

agg["metric_sem"] = agg["metric_std"] / np.sqrt(agg["metric_n"])
agg["epoch_sem"] = agg["epoch_std"] / np.sqrt(agg["epoch_n"])

per_dataset_model = (
    df.groupby(["model_family", "dataset"])
    .agg(
        metric=(METRIC, "mean"),
        train_time=("training_time_s", "mean"),
        n_epochs=("n_epochs", "mean"),
    )
    .reset_index()
)

agg_model = (
    per_dataset_model.groupby("model_family")
    .agg(
        metric_mean=("metric", "mean"),
        metric_std=("metric", "std"),
        metric_n=("metric", "count"),
        time_mean=("train_time", "mean"),
        time_std=("train_time", "std"),
        epochs_mean=("n_epochs", "mean"),
    )
    .reindex(MODEL_ORDER)
)

agg_model["metric_sem"] = agg_model["metric_std"] / np.sqrt(agg_model["metric_n"])
agg_model["time_sem"] = agg_model["time_std"] / np.sqrt(agg_model["metric_n"])

ts2i_present = [t for t in TS2I_ORDER if t in agg.index and not agg.loc[t].isna().all()]
models_present = [
    m for m in MODEL_ORDER if m in agg_model.index and not agg_model.loc[m].isna().all()
]

n_ds = int(agg.loc[ts2i_present[0], "metric_n"])
print(f"\n{'=' * 70}")
print(f"STATISTICS — {METRIC_LABEL} (mean per dataset -> mean across {n_ds} datasets)")
print(f"{'=' * 70}")
print(
    agg[
        [
            "metric_mean",
            "metric_sem",
            "epoch_mean",
            "epoch_sem",
            "stab_mean",
            "stab_std",
            "time_mean",
            "n_epochs_mean",
        ]
    ]
    .round(4)
    .to_string()
)

spiral = agg.loc["SPIRAL"]
print(f"\n{'-' * 70}")
print("KEY COMPARISONS vs SPIRAL:")
print(
    f"{'TS2I':8s}  {METRIC_LABEL + ' diff':>13s}  {'epoch diff':>11s}  {'stab diff':>10s}"
)
print(f"{'-' * 70}")
for t in ts2i_present:
    if t == "SPIRAL":
        continue
    r = agg.loc[t]
    perf = r["metric_mean"] - spiral["metric_mean"]
    ep_p = (
        100 * (r["epoch_mean"] - spiral["epoch_mean"]) / (spiral["epoch_mean"] + 1e-9)
    )
    st_p = 100 * (r["stab_mean"] - spiral["stab_mean"]) / (spiral["stab_mean"] + 1e-9)
    print(f"  {t:8s}  {METRIC_LABEL}:{perf:+.4f}  ep:{ep_p:+7.1f}%  stab:{st_p:+6.1f}%")

print(
    f"\nSPIRAL: {METRIC_LABEL}={spiral['metric_mean']:.4f}±{spiral['metric_sem']:.4f} | "
    f"epoch={spiral['epoch_mean']:.1f}±{spiral['epoch_sem']:.1f} | "
    f"CoV={spiral['stab_mean']:.4f}±{spiral['stab_std']:.4f}"
)

print(f"\n{'=' * 70}")
print("BACKBONE COST vs PERFORMANCE")
print(f"{'=' * 70}")
print(
    agg_model[["metric_mean", "metric_sem", "time_mean", "time_sem", "epochs_mean"]]
    .round(4)
    .to_string()
)

cnn = agg_model.loc["CNN-AE"]
print(f"\nRELATIVE TO CNN-AE:")
for m in models_present:
    if m == "CNN-AE":
        continue
    r = agg_model.loc[m]
    print(
        f"  {m}: {METRIC_LABEL} {r['metric_mean'] - cnn['metric_mean']:+.4f} | "
        f"time {r['time_mean'] / cnn['time_mean']:.2f}x | "
        f"epochs {100 * (r['epochs_mean'] - cnn['epochs_mean']) / cnn['epochs_mean']:+.1f}%"
    )

gap_ts2i = agg["metric_mean"].max() - agg["metric_mean"].min()
gap_backbone = agg_model["metric_mean"].max() - agg_model["metric_mean"].min()
print(
    f"\nGap TS2I: {gap_ts2i:.4f} | Gap backbone: {gap_backbone:.4f} | Ratio: {gap_ts2i / gap_backbone:.1f}x"
)


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


fig, ax = plt.subplots(figsize=(7, 5.5))

for t in ts2i_present:
    r = agg.loc[t]
    is_spiral = t == "SPIRAL"
    col = SPIRAL_COLOR if is_spiral else OTHER_COLOR
    cross_col = SPIRAL_DARK if is_spiral else "#aaaaaa"
    zord = 5 if is_spiral else 2
    lw = 1.8 if is_spiral else 0.9
    ab = 0.85 if is_spiral else 0.45
    size = np.clip(r["stab_mean"] * 1200, 60, 700)

    # ±1 SEM crosshairs
    ax.plot(
        [r["epoch_mean"] - r["epoch_sem"], r["epoch_mean"] + r["epoch_sem"]],
        [r["metric_mean"], r["metric_mean"]],
        color=cross_col,
        lw=lw,
        alpha=ab,
        zorder=zord - 1,
    )
    ax.plot(
        [r["epoch_mean"], r["epoch_mean"]],
        [r["metric_mean"] - r["metric_sem"], r["metric_mean"] + r["metric_sem"]],
        color=cross_col,
        lw=lw,
        alpha=ab,
        zorder=zord - 1,
    )

    ax.scatter(
        r["epoch_mean"],
        r["metric_mean"],
        s=size,
        color=col,
        edgecolors="white",
        lw=1.0,
        alpha=0.92,
        zorder=zord,
    )

    if is_spiral:
        bubble_radius_y = (
            np.sqrt(size) / 2 / 72 * (ax.get_ylim()[1] - ax.get_ylim()[0])
            if ax.get_ylim()[1] != ax.get_ylim()[0]
            else 0.01
        )
        ax.annotate(
            t,
            (r["epoch_mean"], r["metric_mean"]),
            textcoords="offset points",
            xytext=(25, 3),  # SPIRAL text annotation on plot
            fontsize=9.5,
            fontweight="bold",
            color=col,
            ha="center",
        )
    else:
        ax.annotate(
            t,
            (r["epoch_mean"], r["metric_mean"]),
            textcoords="offset points",
            xytext=(6, 3),
            fontsize=8,
            color=col,
        )

ax.axvline(
    agg.loc[ts2i_present, "epoch_mean"].mean(), color="grey", lw=0.7, ls=":", alpha=0.5
)
ax.axhline(
    agg.loc[ts2i_present, "metric_mean"].mean(), color="grey", lw=0.7, ls=":", alpha=0.5
)

ax.set_xlabel("Mean epochs to reach 90% training loss reduction")
ax.set_ylabel(f"Mean {METRIC_LABEL}")
ax.grid(alpha=0.12, ls="--")
plt.tight_layout()

out = f"convergence/{METRIC}/bubble_{METRIC}.pdf"
plt.savefig(out, bbox_inches="tight", dpi=1500)
plt.close()

colors_b = [MODEL_COLORS[m] for m in models_present]

per_dataset_ts2i_model = (
    df.groupby(["ts2i", "model_family", "dataset"])
    .agg(metric=(METRIC, "mean"))
    .reset_index()
)

for t in ts2i_present:
    is_spiral = t == "SPIRAL"
    sub = per_dataset_ts2i_model[per_dataset_ts2i_model["ts2i"] == t]

    fig, ax = plt.subplots(figsize=(5, 4))

    metric_data = [
        sub[sub["model_family"] == m]["metric"].dropna().values for m in models_present
    ]

    parts = ax.violinplot(
        metric_data,
        positions=np.arange(len(models_present)),
        showmedians=True,
        showextrema=True,
        widths=0.55,
    )
    style_violin(parts, colors_b)

    for i, (m, vals) in enumerate(zip(models_present, metric_data)):
        if len(vals) == 0:
            continue
        ax.scatter(
            i,
            np.mean(vals),
            marker="D",
            s=40,
            color=MODEL_COLORS[m],
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

    ax.set_xticks(np.arange(len(models_present)))
    ax.set_xticklabels(models_present)
    ax.set_ylabel(f"Mean {METRIC_LABEL}")
    ax.grid(alpha=0.12, ls="--", axis="y")
    plt.tight_layout()

    out = f"convergence/{METRIC}/ts2i_backbone_{t}_{METRIC}.pdf"
    plt.savefig(out, bbox_inches="tight", dpi=1500)
    plt.close()


per_dataset_ts2i_time = (
    df.groupby(["ts2i", "model_family", "dataset"])
    .agg(train_time=("training_time_s", "mean"))
    .reset_index()
)

for t in ts2i_present:
    is_spiral = t == "SPIRAL"
    sub = per_dataset_ts2i_time[per_dataset_ts2i_time["ts2i"] == t]

    fig, ax = plt.subplots(figsize=(5, 4))

    time_data_t = [
        sub[sub["model_family"] == m]["train_time"].dropna().values
        for m in models_present
    ]

    parts = ax.violinplot(
        time_data_t,
        positions=np.arange(len(models_present)),
        showmedians=True,
        showextrema=True,
        widths=0.55,
    )
    style_violin(parts, colors_b)

    for i, (m, vals) in enumerate(zip(models_present, time_data_t)):
        if len(vals) == 0:
            continue
        ax.scatter(
            i,
            np.mean(vals),
            marker="D",
            s=40,
            color=MODEL_COLORS[m],
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

    ax.set_xticks(np.arange(len(models_present)))
    ax.set_xticklabels(models_present)
    ax.set_ylabel("Mean training time (s)")
    ax.grid(alpha=0.12, ls="--", axis="y")
    plt.tight_layout()

    out = f"convergence/TRAINING_TIME/ts2i_backbone_{t}_training_time_s.pdf"
    plt.savefig(out, bbox_inches="tight", dpi=1500)
    plt.close()

print(f"\n{'=' * 70}")
print(f"PER TS2I × BACKBONE MEANS — {METRIC_LABEL}")
print(f"{'=' * 70}")
for t in ts2i_present:
    sub = per_dataset_ts2i_model[per_dataset_ts2i_model["ts2i"] == t]
    is_spiral = t == "SPIRAL"
    marker = " <- SPIRAL" if is_spiral else ""
    print(f"\n  {t}{marker}")
    for m in models_present:
        vals = sub[sub["model_family"] == m]["metric"].dropna().values
        if len(vals) == 0:
            continue
        print(
            f"    {m:20s}:  mean={np.mean(vals):.4f}  "
            f"median={np.median(vals):.4f}  "
            f"std={np.std(vals):.4f}  n={len(vals)}"
        )

print(f"\n{'=' * 70}")
print("PER TS2I × BACKBONE MEANS — TRAINING TIME (s)")
print(f"{'=' * 70}")
for t in ts2i_present:
    sub = per_dataset_ts2i_time[per_dataset_ts2i_time["ts2i"] == t]
    is_spiral = t == "SPIRAL"
    marker = " <- SPIRAL" if is_spiral else ""
    print(f"\n  {t}{marker}")
    for m in models_present:
        vals = sub[sub["model_family"] == m]["train_time"].dropna().values
        if len(vals) == 0:
            continue
        print(
            f"    {m:20s}:  mean={np.mean(vals):.2f}s  "
            f"median={np.median(vals):.2f}s  "
            f"std={np.std(vals):.2f}s  n={len(vals)}"
        )
