import osmium
import csv

class POIHandler(osmium.SimpleHandler):

    def __init__(self):
        super().__init__()
        self.pois = []
        self.count = 0

    def get_category(self, tags):
        for key in ['amenity', 'shop', 'tourism', 'leisure', 'office', 'craft', 'historic', 'healthcare', 'emergency', 'man_made', 'sport', 'military', 'public_transport']:
            if key in tags:
                return tags[key]
        return ''

    def node(self, n):
        tags = {tag.k: tag.v for tag in n.tags}
        poi_keys = {'amenity', 'shop', 'tourism', 'leisure', 'office', 'craft', 'historic', 'healthcare', 'emergency', 'building', 'landuse', 'natural', 'place', 'highway', 'railway', 'aeroway', 'waterway', 'man_made', 'power', 'barrier', 'boundary', 'sport', 'military', 'diplomatic', 'religion', 'denomination', 'school', 'university', 'industrial', 'commercial', 'residential', 'retail', 'warehouse', 'garages', 'public_transport', 'bus', 'train', 'subway', 'ferry', 'aerialway', 'parking', 'atm', 'bank', 'fuel', 'pharmacy', 'hospital', 'clinic', 'doctors', 'dentist', 'veterinary', 'social_facility', 'restaurant', 'cafe', 'bar', 'fast_food', 'food_court', 'ice_cream', 'pub', 'nightclub', 'museum', 'theatre', 'cinema', 'arts_centre', 'library', 'gallery', 'zoo', 'aquarium', 'theme_park', 'stadium', 'sports_centre', 'hotel', 'hostel', 'motel', 'guest_house', 'camp_site', 'caravan_site', 'chalet', 'information', 'attraction', 'viewpoint', 'picnic_site', 'artwork', 'monument'}
        if not poi_keys.isdisjoint(tags.keys()):
            row = {'name': tags.get('name', ''), 'latitude': n.location.lat, 'longitude': n.location.lon, 'category': self.get_category(tags), 'id': n.id, 'name_ar': tags.get('name:ar', ''), 'name_fr': tags.get('name:fr', ''), 'name_en': tags.get('name:en', ''), 'name_es': tags.get('name:es', ''), 'alt_name': tags.get('alt_name', ''), 'old_name': tags.get('old_name', ''), 'building': tags.get('building', ''), 'building_levels': tags.get('building:levels', ''), 'landuse': tags.get('landuse', ''), 'natural': tags.get('natural', ''), 'place': tags.get('place', ''), 'highway': tags.get('highway', ''), 'railway': tags.get('railway', ''), 'public_transport': tags.get('public_transport', ''), 'bus': tags.get('bus', ''), 'train': tags.get('train', ''), 'aeroway': tags.get('aeroway', ''), 'waterway': tags.get('waterway', ''), 'man_made': tags.get('man_made', ''), 'power': tags.get('power', ''), 'barrier': tags.get('barrier', ''), 'sport': tags.get('sport', ''), 'military': tags.get('military', ''), 'phone': tags.get('phone', ''), 'mobile': tags.get('mobile', ''), 'fax': tags.get('fax', ''), 'email': tags.get('email', ''), 'website': tags.get('website', ''), 'facebook': tags.get('contact:facebook', ''), 'instagram': tags.get('contact:instagram', ''), 'twitter': tags.get('contact:twitter', ''), 'addr_housenumber': tags.get('addr:housenumber', ''), 'addr_street': tags.get('addr:street', ''), 'addr_city': tags.get('addr:city', ''), 'addr_postcode': tags.get('addr:postcode', ''), 'addr_suburb': tags.get('addr:suburb', ''), 'addr_district': tags.get('addr:district', ''), 'addr_province': tags.get('addr:province', ''), 'addr_country': tags.get('addr:country', ''), 'opening_hours': tags.get('opening_hours', ''), 'cuisine': tags.get('cuisine', ''), 'stars': tags.get('stars', ''), 'rooms': tags.get('rooms', ''), 'capacity': tags.get('capacity', ''), 'fee': tags.get('fee', ''), 'charge': tags.get('charge', ''), 'brand': tags.get('brand', ''), 'operator': tags.get('operator', ''), 'network': tags.get('network', ''), 'wheelchair': tags.get('wheelchair', ''), 'wifi': tags.get('wifi', ''), 'air_conditioning': tags.get('air_conditioning', ''), 'outdoor_seating': tags.get('outdoor_seating', ''), 'takeaway': tags.get('takeaway', ''), 'delivery': tags.get('delivery', ''), 'drive_through': tags.get('drive_through', ''), 'religion': tags.get('religion', ''), 'denomination': tags.get('denomination', ''), 'school': tags.get('school', ''), 'isced_level': tags.get('isced:level', ''), 'description': tags.get('description', ''), 'note': tags.get('note', ''), 'wikidata': tags.get('wikidata', ''), 'wikipedia': tags.get('wikipedia', ''), 'image': tags.get('image', ''), 'ref': tags.get('ref', ''), 'source': tags.get('source', '')}
            self.pois.append(row)
            self.count += 1
            if self.count % 10000 == 0:
                print(f'  {self.count} POIs found so far...')
print('Starting full POI extraction from Morocco OSM data...')
handler = POIHandler()
handler.apply_file('morocco-latest.osm.pbf', locations=True)
print(f'\nSaving {len(handler.pois)} POIs to CSV...')
if handler.pois:
    with open('poi.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=handler.pois[0].keys())
        writer.writeheader()
        writer.writerows(handler.pois)
print(f'\n✅ Done! {len(handler.pois)} POIs saved to poi.csv')