import json
import pandas as pd
from pathlib import Path

metadata_dirs = [
    "01_NAB",
    "02_WSD",
    "03_MSL",
    "04_Stock",
    "05_Daphnet",
    "06_MITDB",
    "07_SMD",
    "08_LTDB",
    "09_MGAB",
    "10_SED",
    "11_SVDB",
    "12_TAO",
    "13_IOPS",
    "14_NEK",
    "15_CATSv2",
    "16_TODS",
    "17_Power",
    "18_UCR",
    "19_SMAP",
    "20_SWaT",
    "21_YAHOO",
    "22_Exathlon",
    "23_OPPORTUNITY",
]

all_metadata = []

for metadata_dir_path in metadata_dirs:
    metadata_dir = Path(metadata_dir_path)

    if not metadata_dir.exists():
        continue

    for json_file in sorted(metadata_dir.glob("*_metadata.json")):
        try:
            with open(json_file, "r") as f:
                data = json.load(f)

            metadata_entry = {
                "file_path": data.get("file_path"),
                "dataset": data.get("dataset"),
                "ts2i_transformation_name": data.get("ts2i_transformation_name"),
                "ts2i_transformation_pretty_name": data.get(
                    "ts2i_transformation_pretty_name"
                ),
                "window_len": data.get("window_len"),
                "stride": data.get("stride"),
                "number_of_train_images": data.get("number_of_train_images"),
                "number_of_test_images": data.get("number_of_test_images"),
                "train_ts2i_transformation_wall_time_s": data.get(
                    "train_ts2i_transformation_wall_time_s"
                ),
                "test_ts2i_transformation_wall_time_s": data.get(
                    "test_ts2i_transformation_wall_time_s"
                ),
                "train_ts2i_transformation_time_per_image_ms": data.get(
                    "train_ts2i_transformation_time_per_image_ms"
                ),
                "test_ts2i_transformation_time_per_image_ms": data.get(
                    "test_ts2i_transformation_time_per_image_ms"
                ),
                "train_ts2i_transformation_throughput_images_per_sec": data.get(
                    "train_ts2i_transformation_throughput_images_per_sec"
                ),
                "test_ts2i_transformation_throughput_images_per_sec": data.get(
                    "test_ts2i_transformation_throughput_images_per_sec"
                ),
                "test_timeseries_length": data.get("test_timeseries_length"),
                "n_workers": data.get("n_workers"),
            }

            all_metadata.append(metadata_entry)

        except Exception as e:
            print(e)

df = pd.DataFrame(all_metadata)
output_path = "ts2i_metadata_combined.csv"
df.to_csv(output_path, index=False)
