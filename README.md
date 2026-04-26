# Living Attack Surface Mapper

Welcome to the **MLCS Project**! This project implements an end-to-end Machine Learning pipeline for cybersecurity threat detection, anomaly scoring, and attack surface mapping. It includes a complete data processing workflow, hybrid machine learning models (Isolation Forest, Random Forest, XGBoost), and an interactive frontend dashboard for visualizing threat parameters and simulating attacks.

## Features

* **Data & Preprocessing**: Scripts to generate synthetic, realistic cybersecurity attack surface datasets with logically correlated features (CVSS scores, credential leaks, etc.) and handle missing values/inconsistencies.
* **Hybrid Machine Learning Models**:
  * **Isolation Forest**: Unsupervised anomaly detection.
  * **Random Forest & XGBoost**: Supervised classification for specific threat vectors.
* **Exploratory Data Analysis (EDA)**: Automated scripts to analyze risk distributions, feature correlations, and generate detailed visual plots.
* **Interactive Threat Dashboard**: 
  * Displays risk metrics, anomaly scores, and dynamic visual trends.
  * Attack simulation feature that injects synthetic anomalies in real-time.
  * Dynamic layout with a sleek, modern UI.

## Repository Structure

* `generate_dataset.py`: Generates the simulated attack surface datasets taking base foundation of  `manual_dataset.csv`.
* `preprocess_data.py`: Prepares the dataset for training, handles missing data, and scales features.
* `eda_analysis.py`: Performs EDA and outputs high-quality visualizations into the `eda_plots/` directory.
* `train_models.py`: Trains the Isolation Forest, Random Forest, and XGBoost models. Logs results and saves artifacts.
* `app.py`: Streamlit-based application/dashboard backend logic.
* `helpers.py`: Utility and helper functions for data manipulation and ML scoring.
* `dashboard/`: React/Vite-based frontend application for visualizing risks, trends, and threat simulations.
* `model_results/`: Stores training metrics, ROC curves, feature importances, and confusion matrices.
* `saved_models/`: Local directory containing the `.pkl` model files *(Note: Large model files are `.gitignore`d)*.

## Setup & Installation

### 1. Prerequisites
Ensure you have **Python 3.8+** installed. If you are using the React dashboard, you will also need **Node.js**.

### 2. Python Backend Setup
Clone the repository and install the necessary Python dependencies (like `pandas`, `scikit-learn`, `xgboost`, `streamlit`, etc.):
```bash
git clone https://github.com/arclii/MLCS_Project.git
cd MLCS_Project
# Install dependencies (ensure you have the required libraries installed)
pip install pandas numpy scikit-learn xgboost matplotlib seaborn streamlit plotly
```

### 3. Frontend Dashboard Setup
If you want to run the React/Vite frontend dashboard:
```bash
cd dashboard
npm install
npm run dev
```

## Running the Pipeline

You can run the ML pipeline end-to-end using the following steps:

1. **Generate the Dataset**:
   ```bash
   python generate_dataset.py
   ```
2. **Preprocess the Data**:
   ```bash
   python preprocess_data.py
   ```
3. **Run Exploratory Data Analysis**:
   ```bash
   python eda_analysis.py
   ```
4. **Train the Models**:
   ```bash
   python train_models.py
   ```
5. **Launch the Streamlit App**:
   ```bash
   streamlit run app.py
   ```

## Results & Artifacts

Model artifacts like confusion matrices, ROC curves, and daily risk timeseries are automatically generated in the `model_results/` folder. Detailed EDA visualizations, including correlation heatmaps and risk breakdowns, can be found in `eda_plots/`.

## Contributing

Feel free to fork this project, submit issues, or create pull requests for new features, ML enhancements, or dashboard UI improvements.

---
*Developed by arclii*
