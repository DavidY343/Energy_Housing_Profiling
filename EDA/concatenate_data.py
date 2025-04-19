import pandas as pd
import os

folder_path = "../data_raw/"

dataframes = []

for file in os.listdir(folder_path):
    if file.endswith(".csv"):
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path, dtype={"cups": str}, sep=";")
        dataframes.append(df)

merged_df = pd.concat(dataframes, ignore_index=True)

output_path = "../data/vertical_raw_data.csv"
merged_df.to_csv(output_path, index=False, sep=";")

print("\033[92m✔ File successfully saved as raw_data.csv\033[0m")
