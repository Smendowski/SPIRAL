import pandas as pd

metric_files = {
    "AUC-PR": "uni_mergedTable_AUC-PR.csv",
    "AUC-ROC": "uni_mergedTable_AUC-ROC.csv",
    "VUS-PR": "uni_mergedTable_VUS-PR.csv",
    "VUS-ROC": "uni_mergedTable_VUS-ROC.csv",
    "Standard-F1": "uni_mergedTable_Standard-F1.csv",
    "PA-F1": "uni_mergedTable_PA-F1.csv",
    "Event-based-F1": "uni_mergedTable_Event-based-F1.csv",
    "R-based-F1": "uni_mergedTable_R-based-F1.csv",
    "Affiliation-F1": "uni_mergedTable_Affiliation-F1.csv",
}


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


aggregated_metrics = {}


for metric_name, file_path in metric_files.items():
    print(f"Processing: {metric_name}...")

    df = pd.read_csv(file_path)
    model_cols = [col for col in df.columns if col not in metadata_cols]
    mean_values = df[model_cols].mean()
    aggregated_metrics[metric_name] = mean_values


result_df = pd.DataFrame(aggregated_metrics)
result_df.index.name = "Method"
result_df = result_df.reindex(model_order)

print("\n" + "=" * 80)
print("Aggregation done!")
print("=" * 80)
print(f"\nShape: {result_df.shape}")
print(f"\nFirst rows:\n{result_df.head()}")

print("\n" + "=" * 80)
print("LATEX TABLE ROWS (copy-paste ready):")
print("=" * 80)
print()

for model in result_df.index:
    row_values = result_df.loc[model]
    formatted_values = [f"{val:.2f}" for val in row_values]
    latex_row = f"{model} & " + " & ".join(formatted_values) + " \\\\"
    print(latex_row)
