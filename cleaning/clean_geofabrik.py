import csv
input_file = 'poi.csv'
output_file = 'poi_fixed.csv'
fallback_columns = ['building', 'building_levels', 'landuse', 'natural', 'place', 'highway', 'railway', 'public_transport', 'bus', 'train', 'aeroway', 'waterway', 'man_made', 'power', 'barrier', 'sport', 'military', 'phone', 'mobile', 'fax', 'email', 'website', 'facebook', 'instagram', 'twitter', 'addr_housenumber', 'addr_street', 'addr_city', 'addr_postcode', 'addr_suburb', 'addr_district', 'addr_province', 'addr_country', 'opening_hours', 'cuisine', 'stars', 'rooms', 'capacity', 'fee', 'charge', 'brand', 'operator', 'network', 'wheelchair', 'wifi', 'air_conditioning', 'outdoor_seating', 'takeaway', 'delivery', 'drive_through', 'religion', 'denomination', 'school', 'isced_level', 'description', 'note', 'wikidata', 'wikipedia', 'image', 'ref', 'source']
with open(input_file, 'r', encoding='utf-8') as infile:
    reader = csv.DictReader(infile)
    fieldnames = reader.fieldnames
    rows = []
    for row in reader:
        if not row['category'].strip():
            for col in fallback_columns:
                if row.get(col, '').strip():
                    row['category'] = row[col].strip()
                    break
        rows.append(row)
with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
print(f'✅ Done! {len(rows)} rows written to {output_file}')
import pandas as pd
import numpy as np
df = pd.read_csv('poi_fixed.csv')
df = df.replace('', np.nan)
df_cleaned = df.dropna(subset=['name'])
df_cleaned = df_cleaned.reset_index(drop=True)
df_cleaned.to_csv('cleaned_file.csv', index=False)
df = pd.read_csv('cleaned_file.csv', usecols=['name', 'latitude', 'longitude', 'category'], low_memory=False)
replacement_map = {'platform': 'bus_stop'}
df['category'] = df['category'].replace(replacement_map)
df.to_csv('geofabric_final.csv', index=False)