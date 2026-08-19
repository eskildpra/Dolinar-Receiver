import glob
import pandas as pd

all_files = sorted(glob.glob("runs/*/results.csv"))
df = pd.concat([pd.read_csv(f) for f in all_files], ignore_index=True)
df.to_csv("combined_results.csv", index=False)
df.to_parquet("combined_results.parquet", index=False)

print(f"Successfully merged {len(all_files)} runs into combined_results.csv ({len(df)} total data rows).")