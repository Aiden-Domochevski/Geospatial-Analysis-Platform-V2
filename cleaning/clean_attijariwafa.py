import pandas as pd
df = pd.read_csv('agencies_full.csv')
lat_min, lat_max = (20, 36.0)
lon_min, lon_max = (-17, -1.0)
morocco_branches = df[(df['latitude'] >= lat_min) & (df['latitude'] <= lat_max) & (df['longitude'] >= lon_min) & (df['longitude'] <= lon_max)]
print(morocco_branches.head())
morocco_branches.to_csv('agencies_final.csv', index=False)