# Attijariwafa Bank — Branch Intelligence & Site Selection Platform

This repository contains the source code for the machine learning pipeline and the interactive Streamlit dashboard developed for the territorial intelligence and site selection project at Attijariwafa Bank.

**Note:** For data privacy and security reasons, the datasets (`data/` folder), training notebooks, and the final project documentation (`rapport.tex`, diagrams, figures) have been explicitly excluded from this repository.

## Prerequisites

To run this project, you need:
- **Python 3.14.3** (This exact version is recommended to ensure full compatibility with the generated dependencies).
- A virtual environment (recommended)

## Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone <repository_url>
   cd final_model
   ```

2. **Install Dependencies:**
   Install the required Python packages (Streamlit, Pandas, Folium, Plotly, Scikit-Learn, XGBoost, etc.).
   ```bash
   pip install -r requirements.txt
   ```
   *(If `requirements.txt` is missing, manually install: `pip install streamlit pandas folium plotly scikit-learn xgboost scipy numpy`)*

3. **Data Placement (Crucial Step):**
   Since data is not included in this repository, you must recreate the data structure manually. Create the following folders in the root directory:
   ```
   data/
   ├── inputs/
   │   ├── branches_scored.csv
   │   ├── branch_performance.csv
   │   ├── density.csv
   │   └── merged_data_clean.csv
   └── outputs/
       └── benchmarking_results.json
   ```
   Place the corresponding proprietary CSV files provided by the data engineering team into the `data/inputs/` directory.

4. **Run the Dashboard:**
   Once the data is in place, you can launch the dashboard locally:
   ```bash
   streamlit run app.py
   ```

## Project Structure (Version Controlled)
- `app.py`: The core Streamlit application containing the dashboard interface and geospatial visualization logic.
- `scripts/`: Contains Python scripts for benchmarking, data cleaning, and model evaluation (`run_benchmarks.py`, etc.).
- `.gitignore`: Secures the repository by preventing proprietary data and heavy documentation from being uploaded.
