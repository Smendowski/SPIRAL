import pandas as pd
import re
from pathlib import Path


df = pd.read_csv("ts2i_metadata_combined.csv")


def extract_train_length(file_path):
    if pd.isna(file_path):
        return None
    filename = Path(file_path).stem
    match = re.search(r"tr_(\d+)", filename)
    if match:
        return int(match.group(1))
    return None


df["train_timeseries_length"] = df["file_path"].apply(extract_train_length)

table1 = (
    df.groupby("dataset")
    .agg(
        {
            "file_path": "nunique",
            "train_timeseries_length": "mean",
            "test_timeseries_length": "mean",
            "number_of_train_images": "mean",
            "number_of_test_images": "mean",
            "window_len": "mean",
        }
    )
    .rename(
        columns={
            "file_path": "Files",
            "train_timeseries_length": "Avg Train Length",
            "test_timeseries_length": "Avg Test Length",
            "number_of_train_images": "Avg Train Windows",
            "number_of_test_images": "Avg Test Windows",
            "window_len": "Window Size",
        }
    )
)

table1 = table1.round(0).astype(int)


print("\n" + "=" * 80)
print("LATEX TABLE ROWS:")
print("=" * 80)
print(
    "Dataset & Files & Avg Train Length & Avg Test Length & Avg Train Windows & Avg Test Windows & Window Size \\\\"
)
print("\\hline")
for dataset in table1.index:
    row = table1.loc[dataset]
    latex_row = f"{dataset} & {row['Files']} & {row['Avg Train Length']} & {row['Avg Test Length']} & {row['Avg Train Windows']} & {row['Avg Test Windows']} & {row['Window Size']} \\\\"
    print(latex_row)
