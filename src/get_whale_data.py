import os
import pandas as pd
from src.whales_provider import WhalesDataProvider  # ⬅️ new import



# 2️⃣ Initialize provider
provider = WhalesDataProvider()

# 3️⃣ Your 11 filer IDs (same as before)
filer_ids = [166, 349, 373, 443, 1188, 1909, 261060, 2414, 2764, 3005, 3049]

# 4️⃣ Fetch ALL quarters of holdings for these filers
print("Downloading holdings for all quarters for all filers...")

df = provider.fetch_data(
    "holdings",
    filer_ids=filer_ids,
    all_quarters=1,
    include_13d=0,
)

print(f"Downloaded {len(df)} rows.")

# 5️⃣ Ensure raw data folder exists and save CSV there
os.makedirs("data/raw", exist_ok=True)

output_path = "data/raw/13f_raw_holdings.csv"
df.to_csv(output_path, index=False)

print(f"Saved → {output_path}")
