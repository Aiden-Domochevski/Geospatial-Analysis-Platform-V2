import pandas as pd
df = pd.read_csv('morocco_gmaps_businesses.csv', usecols=['name', 'query', 'latitude', 'longitude'])
df = df.rename(columns={'query': 'category'})
df['category'] = df['category'].str.replace('Ã©', 'e', regex=False)
df.to_csv('google_fina.csv', index=False)
with open('google_fina.csv', 'r') as f:
    bytes = f.read()
with open('google_final.csv', 'w') as s:
    for word in bytes:
        if word == 'Ã':
            s.write('e')
        elif word == '©':
            pass
        else:
            s.write(word)