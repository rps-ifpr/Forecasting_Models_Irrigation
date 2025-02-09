# Forecasting Models for Irrigation Analysis 🌱

This repository contains machine learning models for forecasting irrigation-related time series data. The project implements and compares various deep learning architectures for time series forecasting, utilizing state-of-the-art methods to enhance accuracy and performance.

## 📜 Project History

- **Initial Implementation (`src/`)**
  - Developed core forecasting models.
  - Established basic training and prediction pipelines.
  - Created initial visualization capabilities.
  - Generated first version of model forecasts (v1).

- **Enhanced Implementation (`src_v2/`)**
  - Added modular application structure.
  - Implemented comprehensive model validation.
  - Expanded visualization capabilities.
  - Generated improved model forecasts (v2).
  - Added detailed performance tracking.

## 🗂 Repository Structure

```plaintext
.
├── data/                                      # Historical and forecast data
│   ├── dados1975-2015.csv                     # Main historical dataset (1975-2015)
│   ├── data_est_local.CSV                     # Local station data
│   ├── data.csv                               # Processed dataset
│   ├── generated_data_models.csv              # Generated model data
│   ├── v1/                                    # Version 1 model forecasts
│   └── v2/                                    # Version 2 model forecasts
├── img/                                       # Visualization outputs
│   ├── myplot*.png                            # Various result visualizations
│   ├── myplotcompmodels.png                   # Model comparisons
│   ├── myplotcorrelacao.png                   # Correlation analysis
│   └── myplotmapacalormetricas.png            # Metric heatmaps
├── src/                                       # Original implementation
│   ├── 01_model.py                            # Base model implementation
│   ├── 02_train_model.py                      # Training script
│   ├── 03_predict_test_data.py                # Prediction script
│   ├── 04_mod_Transfor_TFT.py                 # Temporal Fusion Transformer
│   ├── 05_mod_RNN_AutoRNN.py                  # RNN implementation
│   ├── ...                                    # Other model implementations
│   ├── output/                                # Training outputs
│   │   ├── checkpoints/                       # Model checkpoints
│   │   └── lightning_logs/                    # Training logs
│   └── model_metrics_summary.csv              # Model metrics
├── src_v.2/                                   # Enhanced implementation
│   ├── app*.py                                # 17 individual application scripts
│   ├── appv2-1.py                             # Main application entry point
│   ├── app_validation.py                      # Validation scripts
│   ├── consolidated_model_metrics.csv         # Detailed metrics
│   ├── myplot_app*.png                        # Model-specific visualizations
│   └── lightning_logs/                        # Extensive training logs (170+ versions)
├── requirements.txt                           # Python dependencies
└── README.md                                  # This file

```

🛠 Implemented Models
The project includes implementations of:

RNN Variants

Basic RNN
LSTM
GRU
TCN (Temporal Convolutional Network)
BiTCN (Bidirectional TCN)
Dilated RNN
DeepAR
Transformer Variants

Vanilla Transformer
Informer
Autoformer
FEDformer
PatchTST
iTransformer
📊 Data Sources
Historical irrigation data from 1975-2015.
Local station data.
Generated forecast data from multiple models.
Model performance metrics.
🔧 Usage
Install dependencies: