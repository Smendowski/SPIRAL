import pandas as pd

metric = "Affiliation-F1"
df = pd.read_csv(f"uni_mergedTable_{metric}.csv")

df["dataset"] = df["file"].str.extract(r"^\d+_([^_]+)_")[0]

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
model_cols = [col for col in df.columns if col not in metadata_cols]

aggregated_mean = df.groupby("dataset")[model_cols].mean()
aggregated_transposed = aggregated_mean.T

model_order = [
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

print("=" * 80)
print(f"{metric} LATEX TABLE ROWS:")
print("=" * 80)
print()

for model in aggregated_sorted.index:
    row_values = aggregated_sorted.loc[model]
    formatted_values = [f"{val:.2f}" for val in row_values]
    latex_row = f"{model} & " + " & ".join(formatted_values) + " \\\\"
    print(latex_row)
