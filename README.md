# Attijariwafa Bank — Branch Intelligence & Site Selection Platform

This repository contains the source code for the machine learning pipeline and the interactive Streamlit dashboard developed for the territorial intelligence and site selection project at Attijariwafa Bank.

## Project Overview

The objective of this platform is to leverage spatial data, machine learning, and geographic points of interest (POIs) to optimize Attijariwafa Bank's physical branch network. The platform provides a suite of interactive, data-driven tools for:
- **Site Selection**: Predicting the success and expected revenue of a potential new branch location based on surrounding infrastructure and competitor density.
- **Network Optimization**: Automatically identifying redundant, highly overlapping, or underperforming branches that should be merged or closed to maximize overall ROI.
- **Territorial Intelligence**: Segmenting existing branches using ML clustering to understand geographic performance drivers across different cities and regions.

**Note:** For data privacy and size constraints, the actual datasets have been excluded from this repository. You must download them separately via the provided Google Drive link below. The `data/` directory structure itself is preserved for your convenience.

## Prerequisites

To run this project, you need:
- **Python 3.14.3**: You can download this specific version from [here](https://www.npackd.org/p/org.python.Python64/3.14.3). This exact version is recommended to ensure full compatibility with the generated dependencies.
- **Google Chrome**: You can download it from [here](https://www.google.com/chrome/) (required only if you want to use the Google Maps scraper)
- A virtual environment (recommended)

## Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Salaheddineahadaf/Geospatial-Analysis-Platform.git
   cd Geospatial-Analysis-Platform
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Data Placement (Crucial Step):**
   The datasets are not stored directly in this GitHub repository due to file size and privacy constraints. The required folder structure (`data/inputs/` and `data/outputs/`) is already created for you in the repository.
   
   - **Download the Data:** Get the required files from this [Google Drive Link](https://drive.google.com/drive/folders/1IeePZw1w69xwTbbwtLAFhFw98pY6EcfH?usp=sharing).
   - **Extract/Move the Data:** **Important:** ALL the downloaded data files must be placed directly into the pre-existing `data/inputs/` directory. Do not place them in the root `data/` folder or `data/outputs/`.

4. **Run the Dashboard:**
   Once the data is in place, you can launch the dashboard locally:
   ```bash
   streamlit run app.py
   ```
   The dashboard should open automatically in your browser.

## Running the Standalone Scraper

If you only need to run the Google Maps scraper without launching the full dashboard, you can use the standalone version located in the `scripts/` folder:

```bash
streamlit run scripts/scrapper.py
```

This standalone app provides the exact same Data Enrichment interface as the one inside the main dashboard, including:
- Google Maps scraping with resume/fresh-start modes
- Real-time terminal output streaming
- The Clean & Merge pipeline (Step A: clean raw CSV, Step B: merge all data sources and deduplicate)

**Additional scraper dependencies:**
```bash
pip install selenium webdriver-manager colorama
```

## Project Structure

```
Geospatial-Analysis-Platform/
├── app.py                  # Main Streamlit dashboard application
├── logo.png                # Dashboard logo used in the UI
├── requirements.txt        # Python dependencies
├── data/
│   ├── inputs/             # Place the downloaded CSV files here
│   └── outputs/            # Generated output files (created at runtime)
└── scripts/
    └── scrapper.py         # Standalone Google Maps scraper (same UI as the dashboard tab)
```
