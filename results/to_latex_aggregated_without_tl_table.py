import pandas as pd
import re

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


def collapse_name(col):
    col = re.sub(r"ResNet18-AE-(FE|PU|DLR)", "ResNet18-AE", col)
    col = re.sub(r"PVT-v2-b1-AE-(FE|PU|DLR)", "PVT-v2-b1-AE", col)
    return col


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
    "ResNet18-AE (RN)",
    "ResNet18-AE (LP)",
    "ResNet18-AE (SG)",
    "ResNet18-AE (GASF)",
    "ResNet18-AE (GADF)",
    "ResNet18-AE (MTF)",
    "ResNet18-AE (RP)",
    "ResNet18-AE (RWT)",
    "ResNet18-AE (MWT)",
    "ResNet18-AE (SPIRAL)",
    "PVT-v2-b1-AE (RN)",
    "PVT-v2-b1-AE (LP)",
    "PVT-v2-b1-AE (SG)",
    "PVT-v2-b1-AE (GASF)",
    "PVT-v2-b1-AE (GADF)",
    "PVT-v2-b1-AE (MTF)",
    "PVT-v2-b1-AE (RP)",
    "PVT-v2-b1-AE (RWT)",
    "PVT-v2-b1-AE (MWT)",
    "PVT-v2-b1-AE (SPIRAL)",
]

aggregated_metrics = {}

for metric_name, file_path in metric_files.items():
    print(f"Processing: {metric_name}...")
    df = pd.read_csv(file_path)
    model_cols = [col for col in df.columns if col not in metadata_cols]
    rename_map = {col: collapse_name(col) for col in model_cols}

    best_per_combo = {}
    for target in model_order:
        variants = [orig for orig, mapped in rename_map.items() if mapped == target]
        if not variants:
            best_per_combo[target] = float("nan")
            continue
        # pick TL strategy with highest mean across datasets
        best_per_combo[target] = max(df[v].mean() for v in variants)

    aggregated_metrics[metric_name] = pd.Series(best_per_combo)

result_df = pd.DataFrame(aggregated_metrics)
result_df.index.name = "Method"
result_df = result_df.reindex(model_order)

print("\n" + "=" * 80)
print("LATEX TABLE ROWS (copy-paste ready):")
print("=" * 80)

for model in result_df.index:
    row_values = result_df.loc[model]
    formatted_values = [f"{val:.2f}" for val in row_values]
    latex_row = f"{model} & " + " & ".join(formatted_values) + " \\\\"
    print(latex_row)
