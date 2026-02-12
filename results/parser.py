import pandas as pd
import glob

dataset_order = [
    "NAB",
    "WSD",
    "MSL",
    "Stock",
    "Daphnet",
    "MITDB",
    "SMD",
    "LTDB",
    "MGAB",
    "SED",
    "SVDB",
    "TAO",
    "IOPS",
    "NEK",
    "CATSv2",
    "TODS",
    "Power",
    "UCR",
    "SMAP",
    "SWaT",
    "YAHOO",
    "Exathlon",
    "OPPORTUNITY",
]

model_mapping = {
    "cnn_autoencoder": "CNN-AE",
    "resnet18_autoencoder_frozen": "ResNet18-AE-FE",
    "resnet18_autoencoder_progressive_unfreeze": "ResNet18-AE-PU",
    "resnet18_autoencoder_differential_lr": "ResNet18-AE-DLR",
    "pvt_autoencoder_frozen": "PVT-v2-b1-AE-FE",
    "pvt_autoencoder_progressive_unfreeze": "PVT-v2-b1-AE-PU",
    "pvt_autoencoder_differential_lr": "PVT-v2-b1-AE-DLR",
}

ts2i_order = ["RN", "LP", "SG", "GASF", "GADF", "MTF", "RP", "RWT", "MWT", "SPIRAL"]


def get_dataset_name(filename):
    parts = filename.split("_", 1)
    if len(parts) > 1:
        dataset = parts[1].replace(".csv", "").split("-")[0]
        return dataset
    return None


all_files = glob.glob("*.csv")
uni_merged = glob.glob("uni_mergedTable_*.csv")
backfill = glob.glob("00_BACKFILL-*.csv")
all_files = list(set(all_files) - set(uni_merged))
all_files = list(set(all_files) - set(backfill))

sorted_files = []
for dataset in dataset_order:
    matching_files = [f for f in all_files if get_dataset_name(f) == dataset]
    sorted_files.extend(sorted(matching_files))

dfs = []
for file in sorted_files:
    df = pd.read_csv(file)
    dfs.append(df)
    print(f"Reading: {file} ({len(df)} rows)")

combined_df = pd.concat(dfs, ignore_index=True)
combined_df = combined_df.fillna(0)
print(f"\n Total number of rows after concatenation: {len(combined_df)}")

combined_df["model_name"] = combined_df["model"].map(model_mapping)
combined_df["method"] = (
    combined_df["model_name"]
    + " ("
    + combined_df["ts2i_transformation_pretty_name"]
    + ")"
)

# Select the metric for aggregation
metric = "Affiliation_F"

best_results = combined_df.loc[
    combined_df.groupby(["file_path", "method"])[metric].idxmax()
]
print(f"{best_results.shape}")

pivot_df = best_results.pivot_table(index="file_path", columns="method", values=metric)


def method_sort_key(method_name):
    if " (" in method_name:
        model = method_name.split(" (")[0]
        ts2i = method_name.split("(")[1].rstrip(")")
    else:
        return (999, 999)

    model_order = [
        "CNN-AE",
        "ResNet18-AE-FE",
        "ResNet18-AE-PU",
        "ResNet18-AE-DLR",
        "PVT-v2-b1-AE-FE",
        "PVT-v2-b1-AE-PU",
        "PVT-v2-b1-AE-DLR",
    ]
    model_idx = model_order.index(model) if model in model_order else 999

    ts2i_idx = ts2i_order.index(ts2i) if ts2i in ts2i_order else 999

    return (model_idx, ts2i_idx)


sorted_columns = sorted(pivot_df.columns, key=method_sort_key)
pivot_df = pivot_df[sorted_columns]

pivot_df.insert(0, "file", pivot_df.index)

pivot_df.to_csv(f"uni_mergedTable_{metric.replace('_', '-')}.csv", index=False)
