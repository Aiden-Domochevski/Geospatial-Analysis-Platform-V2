# Attijariwafa Bank — Branch Intelligence & Site Selection Platform

This repository contains the source code for the machine learning pipeline and the interactive Streamlit dashboard developed for the territorial intelligence and site selection project at Attijariwafa Bank.

**Note:** For data privacy and size constraints, the actual datasets have been excluded from this repository. You must download them separately via the provided Google Drive link below. The `data/` directory structure itself is preserved for your convenience.

## Prerequisites

To run this project, you need:
- **Python 3.14.3**: You can download this specific version from [here](https://www.npackd.org/p/org.python.Python64/3.14.3). This exact version is recommended to ensure full compatibility with the generated dependencies.
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

## Project Structure

```
Geospatial-Analysis-Platform/
├── app.py                  # Main Streamlit dashboard application
├── logo.png                # Dashboard logo used in the UI
├── requirements.txt        # Python dependencies
├── data/
│   ├── inputs/             # Place the downloaded CSV files here
│   └── outputs/            # Generated output files (created at runtime)
├── scripts/                # Python scripts for benchmarking, data cleaning, and model evaluation
├── notebooks/              # Jupyter notebook for model benchmarking
└── documentation/          # Project report (rapport.tex/pdf), presentation, UML diagrams, and figures
```
