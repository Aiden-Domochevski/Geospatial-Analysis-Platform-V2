import pandas as pd
files = ['google_final.csv', 'osm_final.csv', 'geofabric_final.csv']
df_list = [pd.read_csv(f, encoding='latin1') for f in files]
final_df = pd.concat(df_list, ignore_index=True, sort=False)
final_df.to_csv('merged_data.csv', index=False, encoding='utf-8')
print('Success! Files merged.')
df = pd.read_csv('merged_data.csv')
initial_count = len(df)
final_df = df.drop_duplicates(subset=['latitude', 'longitude'], keep='first')
final_count = len(final_df)
final_df.to_csv('merged_data_clean.csv', index=False, encoding='utf-8')
print(f'Original rows: {initial_count}')
print(f'Rows after removing duplicates: {final_count}')
print(f'Removed {initial_count - final_count} duplicate locations.')