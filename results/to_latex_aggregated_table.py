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
print("LATEX TABLE ROWS (copy-paste ready):")
print("=" * 80)
print()

for model in result_df.index:
    row_values = result_df.loc[model]
    formatted_values = [f"{val:.2f}" for val in row_values]
    latex_row = f"{model} & " + " & ".join(formatted_values) + " \\\\"
    print(latex_row)
