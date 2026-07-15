import csv
import os
import sys
import time
import signal
import random
import threading
from datetime import datetime
from math import cos, radians
try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.common.exceptions import TimeoutException, NoSuchElementException, StaleElementReferenceException
    from webdriver_manager.chrome import ChromeDriverManager
    import pandas as pd
    from colorama import Fore, Back, Style, init
    init(autoreset=True)
except ImportError as e:
    print(f'\n Missing package: {e}')
    print('Run: pip install selenium webdriver-manager pandas colorama')
    sys.exit(1)
OUTPUT_CSV = 'morocco_gmaps_businesses.csv'
CHECKPOINT_FILE = 'scraper_checkpoint.txt'
AUTOSAVE_EVERY = 50
SCROLL_PAUSE = 1.5
MAX_SCROLLS = 15
HEADLESS = True
LAT_MIN, LAT_MAX = (27.6, 35.9)
LON_MIN, LON_MAX = (-13.2, -0.9)
GRID_STEP = 0.3
SEARCH_QUERIES = ['restaurant', 'cafe', 'hotel', 'shop', 'supermarket', 'pharmacy', 'hospital', 'school', 'bank', 'gas station', 'mosque', 'hammam', 'riad', 'salon', 'boutique', 'épicerie', 'boulangerie', 'coiffeur', 'garage', 'dentiste', 'médecin', 'agence immobilière', 'bureau', 'magasin', 'marché', 'souk', 'pâtisserie', 'pizzeria', 'librairie', 'opticien']
scraped_data = []
seen_places = set()
stop_flag = False
lock = threading.Lock()
stats = {'total': 0, 'new': 0, 'skipped': 0, 'cells_done': 0, 'cells_total': 0, 'current_query': '', 'current_city': '', 'start_time': time.time()}

def handle_stop(sig, frame):
    global stop_flag
    stop_flag = True
    print(f'\n\n{Fore.YELLOW}  Stop signal received — saving data and exiting...{Style.RESET_ALL}')
signal.signal(signal.SIGINT, handle_stop)
FIELDNAMES = ['name', 'category', 'rating', 'reviews', 'address', 'phone', 'website', 'latitude', 'longitude', 'city', 'query', 'scraped_at']

def init_csv():
    if not os.path.exists(OUTPUT_CSV):
        with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()

def save_batch(batch):
    with open(OUTPUT_CSV, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writerows(batch)

def load_checkpoint():
    if not os.path.exists(CHECKPOINT_FILE):
        return set()
    with open(CHECKPOINT_FILE, 'r', encoding='utf-8') as f:
        return set((line.strip() for line in f if line.strip()))

def save_checkpoint(key):
    with open(CHECKPOINT_FILE, 'a', encoding='utf-8') as f:
        f.write(key + '\n')

def load_seen_places():
    seen = set()
    if not os.path.exists(OUTPUT_CSV):
        return seen
    try:
        df = pd.read_csv(OUTPUT_CSV, usecols=['name', 'latitude', 'longitude'])
        for _, row in df.iterrows():
            uid = f"{row['name']}_{row['latitude']}_{row['longitude']}"
            seen.add(uid)
        print(f'  {Fore.CYAN}Loaded {len(seen):,} existing records — duplicates will be skipped.{Style.RESET_ALL}')
    except Exception:
        pass
    return seen

def clear_line():
    sys.stdout.write('\x1b[2K\x1b[1G')
    sys.stdout.flush()

def print_header():
    os.system('cls' if os.name == 'nt' else 'clear')
    print(Fore.GREEN + '╔══════════════════════════════════════════════════════════════╗')
    print(Fore.GREEN + '║        🇲🇦  MOROCCO GOOGLE MAPS BUSINESS SCRAPER            ║')
    print(Fore.GREEN + '║         Press  Ctrl+C  to stop and save at any time         ║')
    print(Fore.GREEN + '╚══════════════════════════════════════════════════════════════╝' + Style.RESET_ALL)

def print_stats():
    elapsed = int(time.time() - stats['start_time'])
    hrs, rem = divmod(elapsed, 3600)
    mins, secs = divmod(rem, 60)
    elapsed_str = f'{hrs:02d}:{mins:02d}:{secs:02d}'
    rate = stats['total'] / max(elapsed, 1) * 60
    print(f"\n{Fore.CYAN}{'─' * 64}{Style.RESET_ALL}")
    print(f"  {Fore.WHITE} Total scraped  : {Fore.GREEN}{stats['total']:,}{Style.RESET_ALL}")
    print(f"  {Fore.WHITE} New this run    : {Fore.YELLOW}{stats['new']:,}{Style.RESET_ALL}")
    print(f"  {Fore.WHITE}  Duplicates skip: {Fore.RED}{stats['skipped']:,}{Style.RESET_ALL}")
    print(f"  {Fore.WHITE}  Grid cells     : {Fore.CYAN}{stats['cells_done']}/{stats['cells_total']}{Style.RESET_ALL}")
    print(f'  {Fore.WHITE} Rate           : {Fore.MAGENTA}{rate:.0f} places/min{Style.RESET_ALL}')
    print(f'  {Fore.WHITE}  Elapsed        : {Fore.WHITE}{elapsed_str}{Style.RESET_ALL}')
    print(f"  {Fore.WHITE} Query          : {Fore.BLUE}{stats['current_query']}{Style.RESET_ALL}")
    print(f'  {Fore.WHITE} Saved to       : {Fore.WHITE}{OUTPUT_CSV}{Style.RESET_ALL}')
    print(f"{Fore.CYAN}{'─' * 64}{Style.RESET_ALL}\n")

def make_driver():
    opts = Options()
    if HEADLESS:
        opts.add_argument('--headless=new')
    opts.add_argument('--no-sandbox')
    opts.add_argument('--disable-dev-shm-usage')
    opts.add_argument('--disable-blink-features=AutomationControlled')
    opts.add_argument('--window-size=1280,900')
    opts.add_argument('--lang=en-US')
    opts.add_experimental_option('excludeSwitches', ['enable-automation'])
    opts.add_experimental_option('useAutomationExtension', False)
    opts.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36')
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=opts)
    driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
    return driver
MOROCCO_CITIES = [((33.5, 34.1), (-7.8, -7.4), 'Casablanca'), ((33.9, 34.1), (-6.9, -6.7), 'Rabat'), ((34.0, 34.2), (-5.1, -4.9), 'Fès'), ((33.8, 34.1), (-5.5, -5.2), 'Meknès'), ((31.6, 31.8), (-8.1, -7.9), 'Marrakech'), ((35.1, 35.3), (-6.0, -5.8), 'Tanger'), ((34.6, 34.7), (-1.9, -1.8), 'Oujda'), ((35.1, 35.3), (-3.0, -2.8), 'Nador'), ((30.4, 30.6), (-9.7, -9.5), 'Agadir'), ((32.3, 32.4), (-9.3, -9.1), 'El Jadida'), ((33.0, 33.1), (-8.6, -8.4), 'Settat'), ((34.3, 34.5), (-4.0, -3.8), 'Taza'), ((29.9, 30.0), (-9.6, -9.4), 'Tiznit'), ((32.7, 32.9), (-6.6, -6.4), 'Beni Mellal'), ((32.5, 32.7), (-7.6, -7.4), 'Khouribga')]

def get_city(lat, lon):
    for lat_r, lon_r, city in [(r[0], r[1], r[2]) for r in MOROCCO_CITIES]:
        if lat_r[0] <= lat <= lat_r[1] and lon_r[0] <= lon <= lon_r[1]:
            return city
    return 'Morocco'

def scrape_search(driver, query, lat, lon, city):
    global stop_flag
    batch = []
    url = f"https://www.google.com/maps/search/{query.replace(' ', '+')}/@{lat},{lon},13z"
    try:
        driver.get(url)
        time.sleep(2.5)
        try:
            btn = driver.find_element(By.XPATH, '//button[contains(., "Accept") or contains(., "Accepter")]')
            btn.click()
            time.sleep(1)
        except Exception:
            pass
        wait = WebDriverWait(driver, 8)
        try:
            panel = wait.until(EC.presence_of_element_located((By.XPATH, '//div[@role="feed"]')))
        except TimeoutException:
            return []
        for _ in range(MAX_SCROLLS):
            if stop_flag:
                break
            driver.execute_script('arguments[0].scrollTop += 800', panel)
            time.sleep(SCROLL_PAUSE)
        cards = driver.find_elements(By.XPATH, '//div[@role="feed"]//a[@href]')
        urls = list({c.get_attribute('href') for c in cards if '/maps/place/' in (c.get_attribute('href') or '')})
        for place_url in urls:
            if stop_flag:
                break
            uid = place_url.split('/maps/place/')[-1].split('/')[0]
            if uid in seen_places:
                with lock:
                    stats['skipped'] += 1
                continue
            try:
                driver.get(place_url)
                time.sleep(1.8 + random.uniform(0, 0.5))
                try:
                    name = driver.find_element(By.XPATH, '//h1[contains(@class,"fontHeadlineLarge")]').text.strip()
                except Exception:
                    name = uid.replace('%20', ' ')
                if not name:
                    continue
                try:
                    category = driver.find_element(By.XPATH, '//button[@jsaction and contains(@class,"fontBodyMedium")]').text.strip()
                except Exception:
                    category = ''
                try:
                    rating = driver.find_element(By.XPATH, '//div[@jsaction]//span[@aria-hidden="true"][contains(text(),".")]').text.strip()
                except Exception:
                    rating = ''
                try:
                    reviews = driver.find_element(By.XPATH, '//span[contains(text(),"(") and contains(text(),")")]').text.strip().replace('(', '').replace(')', '').replace(',', '')
                except Exception:
                    reviews = ''
                try:
                    address = driver.find_element(By.XPATH, '//button[@data-item-id="address"]//div[contains(@class,"fontBodyMedium")]').text.strip()
                except Exception:
                    address = ''
                try:
                    phone = driver.find_element(By.XPATH, '//button[starts-with(@data-item-id,"phone")]//div[contains(@class,"fontBodyMedium")]').text.strip()
                except Exception:
                    phone = ''
                try:
                    website = driver.find_element(By.XPATH, '//a[@data-item-id="authority"]').get_attribute('href')
                except Exception:
                    website = ''
                p_lat = p_lon = ''
                try:
                    current_url = driver.current_url
                    if '/@' in current_url:
                        coords = current_url.split('/@')[1].split(',')
                        p_lat = coords[0]
                        p_lon = coords[1]
                    elif '!3d' in current_url:
                        import re
                        m = re.search('!3d(-?\\d+\\.\\d+)!4d(-?\\d+\\.\\d+)', current_url)
                        if m:
                            p_lat = m.group(1)
                            p_lon = m.group(2)
                except Exception:
                    p_lat = str(round(lat, 6))
                    p_lon = str(round(lon, 6))
                record = {'name': name, 'category': category, 'rating': rating, 'reviews': reviews, 'address': address, 'phone': phone, 'website': website, 'latitude': p_lat, 'longitude': p_lon, 'city': city, 'query': query, 'scraped_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                seen_places.add(uid)
                batch.append(record)
                with lock:
                    scraped_data.append(record)
                    stats['total'] += 1
                    stats['new'] += 1
                    stats['current_query'] = f'{query} @ {city}'
                sys.stdout.write(f"\r  {Fore.GREEN} {stats['total']:,} scraped{Style.RESET_ALL}  |  {Fore.YELLOW}  {stats['skipped']:,} dupes{Style.RESET_ALL}  |  {Fore.CYAN}{name[:40]:<40}{Style.RESET_ALL}")
                sys.stdout.flush()
                if stats['total'] % AUTOSAVE_EVERY == 0:
                    save_batch(batch)
                    batch = []
            except Exception as e:
                continue
        if batch:
            save_batch(batch)
            batch = []
        return True
    except Exception as e:
        return []

def build_grid():
    grid = []
    lat = LAT_MIN
    while lat < LAT_MAX:
        lon = LON_MIN
        while lon < LON_MAX:
            grid.append((round(lat + GRID_STEP / 2, 4), round(lon + GRID_STEP / 2, 4)))
            lon += GRID_STEP
        lat += GRID_STEP
    return grid

def main():
    global stop_flag, seen_places
    init_csv()
    print_header()
    completed = load_checkpoint()
    seen_places = load_seen_places()
    if completed:
        print(f'  {Fore.GREEN}  Resuming — {len(completed):,} cell/query combos already done.{Style.RESET_ALL}')
    else:
        print(f'  {Fore.WHITE} Fresh start — no checkpoint found.{Style.RESET_ALL}')
    grid = build_grid()
    stats['cells_total'] = len(grid) * len(SEARCH_QUERIES)
    stats['cells_done'] = len(completed)
    print(f'\n  {Fore.CYAN}Grid cells    : {len(grid):,}{Style.RESET_ALL}')
    print(f'  {Fore.CYAN}Queries/cell  : {len(SEARCH_QUERIES)}{Style.RESET_ALL}')
    print(f"  {Fore.CYAN}Total searches: {stats['cells_total']:,}{Style.RESET_ALL}")
    print(f'  {Fore.CYAN}Already done  : {len(completed):,}{Style.RESET_ALL}')
    print(f'  {Fore.YELLOW}Output file   : {OUTPUT_CSV}{Style.RESET_ALL}')
    print(f'\n  {Fore.WHITE}Starting in 3 seconds — press Ctrl+C anytime to stop & save...{Style.RESET_ALL}\n')
    time.sleep(3)
    driver = make_driver()
    try:
        for lat, lon in grid:
            if stop_flag:
                break
            city = get_city(lat, lon)
            for query in SEARCH_QUERIES:
                if stop_flag:
                    break
                cell_key = f'{lat}|{lon}|{query}'
                if cell_key in completed:
                    continue
                stats['cells_done'] += 1
                stats['current_query'] = f'{query} @ {city}'
                if stats['cells_done'] % 5 == 1:
                    print_header()
                    print_stats()
                scrape_search(driver, query, lat, lon, city)
                save_checkpoint(cell_key)
                completed.add(cell_key)
                time.sleep(random.uniform(1.0, 2.5))
    except Exception as e:
        print(f'\n{Fore.RED}Error: {e}{Style.RESET_ALL}')
    finally:
        driver.quit()
        print_header()
        print(f"\n{Fore.GREEN}{'═' * 64}{Style.RESET_ALL}")
        print(f'{Fore.GREEN}   SCRAPING COMPLETE / STOPPED{Style.RESET_ALL}')
        print(f"{Fore.GREEN}{'═' * 64}{Style.RESET_ALL}")
        print(f"\n  {Fore.WHITE} Total businesses scraped : {Fore.GREEN}{stats['total']:,}{Style.RESET_ALL}")
        print(f'  {Fore.WHITE} Saved to                 : {Fore.YELLOW}{OUTPUT_CSV}{Style.RESET_ALL}')
        elapsed = int(time.time() - stats['start_time'])
        hrs, rem = divmod(elapsed, 3600)
        mins, secs = divmod(rem, 60)
        print(f'  {Fore.WHITE}  Total time               : {Fore.WHITE}{hrs:02d}:{mins:02d}:{secs:02d}{Style.RESET_ALL}')
        print(f'\n  Open {Fore.CYAN}{OUTPUT_CSV}{Style.RESET_ALL} for your data!\n')
if __name__ == '__main__':
    main()