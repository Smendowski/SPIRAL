import pandas as pd

df = pd.read_csv("ts2i_metadata_combined.csv")

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

ts2i_order = ["RN", "LP", "SG", "GASF", "GADF", "MTF", "RP", "RWT", "MWT", "SPIRAL"]

table2 = (
    df.groupby(["ts2i_transformation_pretty_name", "dataset"])[
        "test_ts2i_transformation_throughput_images_per_sec"
    ]
    .mean()
    .unstack(fill_value=0)
)


table2 = table2.reindex(ts2i_order)
existing_datasets = [d for d in dataset_order if d in table2.columns]
table2 = table2[existing_datasets]


table2 = table2.round(2)

table2.to_csv("ts2i_throughput_per_dataset.csv")

print("\n" + "=" * 80)
print("LATEX TABLE ROWS:")
print("=" * 80)
for ts2i in table2.index:
    row_values = table2.loc[ts2i]
    formatted_values = [f"{val:.2f}" for val in row_values]
    latex_row = f"{ts2i} & " + " & ".join(formatted_values) + " \\\\"
    print(latex_row)
