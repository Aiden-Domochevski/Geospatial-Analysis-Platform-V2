import geopandas as gpd
import pandas as pd
gdf = gpd.read_parquet('morocco_places.geoparquet')
gdf.to_csv('morocco_overture_poi.csv', index=False)
print(f'Total POIs: {len(gdf)}')