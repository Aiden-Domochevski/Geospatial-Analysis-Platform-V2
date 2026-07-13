# Attijariwafa Bank — Branch Intelligence & Site Selection Platform

This repository contains the source code for the machine learning pipeline and the interactive Streamlit dashboard developed for the territorial intelligence and site selection project at Attijariwafa Bank.

**Note:** For data privacy and size constraints, the actual datasets have been excluded from this repository. You must download them separately via the provided Google Drive link. The `data/` directory structure itself is preserved for your convenience.

## Prerequisites

To run this project, you need:
- **Python 3.14.3**: You can download this specific version from [here](https://www.npackd.org/p/org.python.Python64/3.14.3). This exact version is recommended to ensure full compatibility with the generated dependencies.
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
   The datasets are not stored directly in this GitHub repository due to file size and privacy constraints. The required folder structure (`data/inputs/` and `data/outputs/`) is already created for you in the repository.
   
   - **Download the Data:** Get the required files from this [Google Drive Link](https://drive.google.com/drive/folders/1IeePZw1w69xwTbbwtLAFhFw98pY6EcfH?usp=sharing).
   - **Extract/Move the Data:** **Important:** ALL the downloaded data files must be placed directly into the pre-existing `data/inputs/` directory. Do not place them in the root `data/` folder or `data/outputs/`.

4. **Run the Dashboard:**
   Once the data is in place, you can launch the dashboard locally:
   ```bash
   streamlit run app.py
   ```

## Project Structure (Version Controlled)
- `app.py`: The core Streamlit application containing the dashboard interface and geospatial visualization logic.
- `scripts/`: Contains Python scripts for benchmarking, data cleaning, and model evaluation (`run_benchmarks.py`, etc.).
- `.gitignore`: Secures the repository by preventing proprietary data and heavy documentation from being uploaded.
