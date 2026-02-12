import pandas as pd

# Choose the metric to be processed
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

aggregated_sorted = aggregated_transposed.reindex(model_order)

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
