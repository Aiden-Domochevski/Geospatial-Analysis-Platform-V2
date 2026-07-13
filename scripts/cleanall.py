import pandas as pd

# 1. List your files
files = ["../../allinone/google_final.csv", "../../allinone/osm_final.csv", "../../allinone/geofabric_final.csv"]

# Use encoding='latin1' to handle the '0xe9' (é) error
df_list = [pd.read_csv(f, encoding='latin1') for f in files]

# Merge them
final_df = pd.concat(df_list, ignore_index=True, sort=False)

# Save the final result as UTF-8 (the modern standard)
final_df.to_csv("../data/inputs/merged_data.csv", index=False, encoding='utf-8')

print("Success! Files merged.")


df=pd.read_csv("../data/inputs/merged_data.csv")

initial_count = len(df)
final_df = df.drop_duplicates(subset=['latitude', 'longitude'], keep='first')
final_count = len(final_df)

# 5. Save the clean result
final_df.to_csv("../data/inputs/merged_data_clean.csv", index=False, encoding='utf-8')

print(f"Original rows: {initial_count}")
print(f"Rows after removing duplicates: {final_count}")
print(f"Removed {initial_count - final_count} duplicate locations.")
