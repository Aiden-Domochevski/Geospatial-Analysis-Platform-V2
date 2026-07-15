import pandas as pd
df = pd.read_csv('morocco_overture_poi.csv')
df = df[df['addresses'].str.contains("'country': 'MA'", na=False)]
df = df.rename(columns={'names': 'name'})
df.to_csv('filtered_file.csv', index=False)
df = pd.read_csv('filtered_file.csv', usecols=['name', 'geometry', 'basic_category'])
df[['longitude', 'latitude']] = df['geometry'].str.extract('\\((?P<lon>[^\\s]+)\\s+(?P<lat>[^\\s]+)\\)')
df['name'] = df['name'].str.split("'primary': '").str[1].str.split("'").str[0]
df = df.rename(columns={'basic_category': 'category'})
df = df.drop(columns=['geometry'])
df = df[['name', 'latitude', 'longitude', 'category']]
df.to_csv('osm_final.csv', index=False)
print(df.head())