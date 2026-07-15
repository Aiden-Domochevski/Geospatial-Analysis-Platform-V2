import streamlit as st
import os, re, time, subprocess, threading, queue
st.set_page_config(page_title='Data Enrichment — Google Maps Scraper', page_icon='🛰️', layout='wide', initial_sidebar_state='collapsed')
st.markdown('<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet">', unsafe_allow_html=True)
st.markdown('\n<style>\n:root {\n    --bg-dark: #0E1117;\n    --card-bg: rgba(28, 32, 40, 0.85);\n    --accent: #CC1B1B;\n    --text-primary: #F0F0F0;\n    --text-muted: #888;\n}\n[data-testid="stAppViewContainer"] {\n    font-family: \'DM Sans\', sans-serif;\n}\nh1, h2, h3 {\n    font-family: \'Syne\', sans-serif !important;\n}\n</style>\n', unsafe_allow_html=True)

def _strip_ansi(text):
    return re.sub('\\x1b\\[[0-9;]*[a-zA-Z]', '', text)

def _stream_reader(proc, output_queue):
    try:
        for line in iter(proc.stdout.readline, ''):
            if line:
                output_queue.put(line)
            else:
                break
    except Exception:
        pass
    finally:
        output_queue.put(None)
st.markdown('\n<div style="font-family:\'Syne\',sans-serif; font-size:1.2rem; font-weight:700;\n            color:#F0F0F0; letter-spacing:0.05em; margin-bottom:0.4rem;">\n     🛰️ Data Enrichment — Google Maps Scraper\n</div>\n<div style="color:#888; font-size:0.9rem; margin-bottom:1.5rem;">\n     Enrich the POI dataset by scraping new data from Google Maps.\n     The output below mirrors exactly what you would see in the terminal.\n</div>\n', unsafe_allow_html=True)
SCRAPER_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'google-maps-checkpoints'))
MAPS_SCRIPT = os.path.join(SCRAPER_DIR, 'maps.py')
CLEAN_SCRIPT = os.path.join(SCRAPER_DIR, 'clean.py')
CHECKPOINT_F = os.path.join(SCRAPER_DIR, 'scraper_checkpoint.txt')
RAW_CSV = os.path.join(SCRAPER_DIR, 'morocco_gmaps_businesses.csv')
CLEANALL_SCRIPT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'allinone', 'cleanall.py'))
if not os.path.exists(MAPS_SCRIPT):
    st.error(f'Scraper not found at `{MAPS_SCRIPT}`')
    st.stop()
col_a, col_b, col_c = st.columns(3)
ckpt_exists = os.path.exists(CHECKPOINT_F)
csv_exists = os.path.exists(RAW_CSV)
with col_a:
    if ckpt_exists:
        with open(CHECKPOINT_F, 'r') as _f:
            ckpt_count = sum((1 for _ in _f))
        st.metric('Checkpoint Entries', f'{ckpt_count:,}')
    else:
        st.metric('Checkpoint Entries', '0')
with col_b:
    if csv_exists:
        try:
            row_count = sum((1 for _ in open(RAW_CSV, encoding='utf-8'))) - 1
            st.metric('Scraped Records', f'{row_count:,}')
        except Exception:
            st.metric('Scraped Records', '—')
    else:
        st.metric('Scraped Records', '0')
with col_c:
    merged_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'inputs', 'merged_data_clean.csv')
    if os.path.exists(merged_path):
        try:
            mr = sum((1 for _ in open(merged_path, encoding='utf-8'))) - 1
            st.metric('Merged POI Total', f'{mr:,}')
        except Exception:
            st.metric('Merged POI Total', '—')
    else:
        st.metric('Merged POI Total', '—')
st.markdown('---')
with st.expander('⚠️ Requirements (click to expand)'):
    st.markdown('\n    The scraper requires the following to be installed:\n    ```\n    pip install selenium webdriver-manager pandas colorama\n    ```\n    You also need **Google Chrome** installed on this machine.\n    The scraper runs in headless mode by default.\n    ')
st.markdown('### Scraping Configuration')
mode = st.radio('Select scraping mode:', [' Resume from Checkpoints', ' Start Fresh (delete existing data)'], index=0, key='scrape_mode', horizontal=True)
col1, col2 = st.columns(2)
start_clicked = col1.button(' Start Scraping', type='primary', use_container_width=True, key='btn_start_scrape')
stop_clicked = col2.button(' Stop Scraper', type='secondary', use_container_width=True, key='btn_stop_scrape')
if 'scraper_process' not in st.session_state:
    st.session_state.scraper_process = None
if 'scraper_log' not in st.session_state:
    st.session_state.scraper_log = ''
if 'scraper_running' not in st.session_state:
    st.session_state.scraper_running = False
if stop_clicked and st.session_state.scraper_process is not None:
    try:
        st.session_state.scraper_process.terminate()
        st.session_state.scraper_process.wait(timeout=5)
    except Exception:
        try:
            st.session_state.scraper_process.kill()
        except Exception:
            pass
    st.session_state.scraper_running = False
    st.session_state.scraper_log += '\n\n=== SCRAPER STOPPED BY USER ==='
    st.toast('Scraper stopped. Data saved up to the last checkpoint.', icon='')
    st.rerun()
if start_clicked:
    if 'Fresh' in mode:
        for fpath in [CHECKPOINT_F, RAW_CSV]:
            if os.path.exists(fpath):
                os.remove(fpath)
        st.session_state.scraper_log = '=== FRESH START — Checkpoints and CSV deleted ===\n\n'
    else:
        st.session_state.scraper_log = '=== RESUMING FROM CHECKPOINTS ===\n\n'
    try:
        scraper_env = os.environ.copy()
        scraper_env['PYTHONIOENCODING'] = 'utf-8'
        proc = subprocess.Popen(['python', '-u', MAPS_SCRIPT], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace', cwd=SCRAPER_DIR, bufsize=1, env=scraper_env)
        st.session_state.scraper_process = proc
        st.session_state.scraper_running = True
        output_queue = queue.Queue()
        reader_thread = threading.Thread(target=_stream_reader, args=(proc, output_queue), daemon=True)
        reader_thread.start()
        terminal_placeholder = st.empty()
        status_placeholder = st.empty()
        while True:
            lines_this_cycle = []
            try:
                while True:
                    item = output_queue.get_nowait()
                    if item is None:
                        st.session_state.scraper_running = False
                        break
                    lines_this_cycle.append(item)
            except queue.Empty:
                pass
            if lines_this_cycle:
                cleaned = _strip_ansi(''.join(lines_this_cycle))
                st.session_state.scraper_log += cleaned
            display_lines = st.session_state.scraper_log.split('\n')
            if len(display_lines) > 200:
                display_lines = display_lines[-200:]
            terminal_placeholder.code('\n'.join(display_lines), language='bash')
            if proc.poll() is not None and (not st.session_state.scraper_running):
                break
            if proc.poll() is not None:
                remaining = proc.stdout.read()
                if remaining:
                    st.session_state.scraper_log += _strip_ansi(remaining)
                st.session_state.scraper_running = False
                break
            time.sleep(0.5)
        display_lines = st.session_state.scraper_log.split('\n')
        if len(display_lines) > 200:
            display_lines = display_lines[-200:]
        terminal_placeholder.code('\n'.join(display_lines), language='bash')
        status_placeholder.success('Scraping session ended. Data has been saved.')
        st.session_state.scraper_process = None
    except FileNotFoundError:
        st.error('Python interpreter not found. Make sure Python is in your PATH.')
    except Exception as e:
        st.error(f'Failed to start the scraper: {e}')
if not start_clicked and st.session_state.scraper_log:
    st.markdown('###  Last Scraping Session Output')
    display_lines = st.session_state.scraper_log.split('\n')
    if len(display_lines) > 200:
        display_lines = display_lines[-200:]
    st.code('\n'.join(display_lines), language='bash')
st.markdown('---')
st.markdown('### Clean & Merge Pipeline')
st.markdown('\n<div style="color:#aaa; font-size:0.85rem; margin-bottom:1rem;">\n    After scraping, run the pipeline below to:<br/>\n    <b>Step A</b> — Clean the raw Google Maps CSV (fix encoding, select columns).<br/>\n    <b>Step B</b> — Merge all 3 data sources (Google Maps + OSM + Geofabrik) and deduplicate.\n</div>\n', unsafe_allow_html=True)
if 'pipeline_log' not in st.session_state:
    st.session_state.pipeline_log = ''
if st.button(' Run Clean & Merge Pipeline', type='primary', use_container_width=True, key='btn_clean_merge'):
    pipeline_output = []
    pipeline_output.append('=' * 60)
    pipeline_output.append('STEP A: Cleaning Google Maps data (clean.py)')
    pipeline_output.append('=' * 60)
    try:
        result_a = subprocess.run(['python', CLEAN_SCRIPT], capture_output=True, text=True, encoding='utf-8', errors='replace', cwd=SCRAPER_DIR, timeout=120)
        pipeline_output.append(result_a.stdout if result_a.stdout else '(no output)')
        if result_a.stderr:
            pipeline_output.append(f'STDERR: {result_a.stderr}')
        if result_a.returncode == 0:
            pipeline_output.append(' clean.py completed successfully.')
        else:
            pipeline_output.append(f' clean.py failed (exit code {result_a.returncode}).')
    except Exception as e:
        pipeline_output.append(f' Error running clean.py: {e}')
    pipeline_output.append('')
    src_google = os.path.join(SCRAPER_DIR, 'google_final.csv')
    dst_google = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'allinone', 'google_final.csv'))
    if os.path.exists(src_google):
        import shutil
        try:
            shutil.copy2(src_google, dst_google)
            pipeline_output.append(f'📋 Copied google_final.csv to allinone/')
        except Exception as e:
            pipeline_output.append(f' Could not copy to allinone: {e}')
    pipeline_output.append('')
    pipeline_output.append('=' * 60)
    pipeline_output.append('STEP B: Merging all sources & deduplicating (cleanall.py)')
    pipeline_output.append('=' * 60)
    try:
        result_b = subprocess.run(['python', CLEANALL_SCRIPT], capture_output=True, text=True, encoding='utf-8', errors='replace', cwd=os.path.dirname(CLEANALL_SCRIPT), timeout=300)
        pipeline_output.append(result_b.stdout if result_b.stdout else '(no output)')
        if result_b.stderr:
            pipeline_output.append(f'STDERR: {result_b.stderr}')
        if result_b.returncode == 0:
            pipeline_output.append(' cleanall.py completed successfully.')
        else:
            pipeline_output.append(f' cleanall.py failed (exit code {result_b.returncode}).')
    except Exception as e:
        pipeline_output.append(f' Error running cleanall.py: {e}')
    pipeline_output.append('')
    pipeline_output.append('=' * 60)
    pipeline_output.append('PIPELINE COMPLETE')
    pipeline_output.append('=' * 60)
    st.session_state.pipeline_log = '\n'.join(pipeline_output)
    st.rerun()
if st.session_state.pipeline_log:
    st.code(st.session_state.pipeline_log, language='bash')