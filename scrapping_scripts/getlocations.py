import requests
import csv
import time
from bs4 import BeautifulSoup
BASE_URL = 'https://backend.attijarimdm.com/fr/api/locator_entity/vactory_locator'
LIMIT = 50

def fetch_page(offset):
    params = {'fields[locator_entity--vactory_locator]': 'name,field_locator_info,field_locator_category,field_locator_description,field_locator_additional_adress,field_locator_phone', 'page[offset]': offset, 'page[limit]': LIMIT}
    headers = {'User-Agent': 'Mozilla/5.0', 'Accept': 'application/vnd.api+json'}
    response = requests.get(BASE_URL, params=params, headers=headers)
    if response.status_code != 200:
        print(f'Error {response.status_code} at offset {offset}')
        return None
    return response.json()

def get_text_from_html(html):
    if not html:
        return ''
    soup = BeautifulSoup(html, 'html.parser')
    return soup.get_text(' ', strip=True)
all_rows = []
offset = 0
while True:
    print(f'Fetching offset {offset} ...')
    data = fetch_page(offset)
    if not data or 'data' not in data or len(data['data']) == 0:
        break
    for item in data['data']:
        attrs = item.get('attributes', {})
        name = attrs.get('name', '')
        info = attrs.get('field_locator_info') or {}
        lat = info.get('lat', '')
        lon = info.get('lon', '')
        addr_html = ''
        addr_block = attrs.get('field_locator_additional_adress')
        if isinstance(addr_block, dict):
            addr_html = addr_block.get('value', '')
        address_text = get_text_from_html(addr_html)
        phone = ''
        phone_block = attrs.get('field_locator_phone')
        if isinstance(phone_block, dict):
            phone = phone_block.get('value', '') or phone_block.get('number', '')
        desc = ''
        desc_block = attrs.get('field_locator_description')
        if isinstance(desc_block, dict):
            desc = desc_block.get('value', '')
        cat = ''
        cat_data = attrs.get('field_locator_category')
        if isinstance(cat_data, dict):
            cat = cat_data.get('id', '')
        all_rows.append({'name': name, 'latitude': lat, 'longitude': lon, 'address': address_text, 'phone': phone, 'description': desc, 'category_id': cat})
    offset += LIMIT
    time.sleep(0.2)
print('Saving to CSV...')
with open('agencies_full.csv', 'w', newline='', encoding='utf-8') as f:
    fieldnames = ['name', 'latitude', 'longitude', 'address', 'phone', 'description', 'category_id']
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(all_rows)
print('Done. Total rows:', len(all_rows))