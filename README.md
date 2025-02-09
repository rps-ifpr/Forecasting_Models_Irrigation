# Forecasting Models for Irrigation Analysis

This repository contains machine learning models for forecasting irrigation-related time series data. The project implements and compares various deep learning architectures for time series forecasting.

## Project History

- **Initial Implementation (src/)**
  - Developed core forecasting models
  - Established basic training and prediction pipelines
  - Created initial visualization capabilities
  - Generated first version of model forecasts (v1)

- **Enhanced Implementation (src_v.2/)**
  - Added modular application structure
  - Implemented comprehensive model validation
  - Expanded visualization capabilities
  - Generated improved model forecasts (v2)
  - Added detailed performance tracking

## Repository Structure

<<<<<<< Updated upstream
With the increasing demand for accurate weather forecasts, this project explores the use of neural networks, especially Transformer models and RNNs, to capture complex patterns in meteorological time series. Fourteen models were evaluated using the [NeuralForecast](https://github.com/Nixtla/neuralforecast) library, employing metrics such as RMSE, RMSPE, Max Abs Error, Mean Abs Error, and Median Abs Error.

The project structure is organized as described below.


## Directory Structure

=======
>>>>>>> Stashed changes
```
.
├── data/                   # Historical and forecast data
│   ├── dados1975-2015.csv  # Main historical dataset (1975-2015)
<<<<<<< Updated upstream
│   ├── data_est_local.CSV  # Local station data - dataset
=======
│   ├── data_est_local.CSV  # Local station data
>>>>>>> Stashed changes
│   ├── data.csv            # Processed dataset
│   ├── generated_data_models.csv  # Generated model data
│   ├── v1/                 # Version 1 model forecasts
│   └── v2/                 # Version 2 model forecasts
├── img/                    # Visualization outputs
│   ├── myplot*.png         # Various result visualizations
│   ├── myplotcompmodels.png  # Model comparisons
│   ├── myplotcorrelacao.png  # Correlation analysis
│   └── myplotmapacalormetricas.png  # Metric heatmaps
├── src/                    # Original implementation
│   ├── 01_model.py         # Base model implementation
│   ├── 02_train_model.py   # Training script
│   ├── 03_predict_test_data.py  # Prediction script
│   ├── 04_mod_Transfor_TFT.py   # Temporal Fusion Transformer
│   ├── 05_mod_RNN_AutoRNN.py    # RNN implementation
│   ├── ...                 # Other model implementations
│   ├── output/             # Training outputs
│   │   ├── checkpoints/    # Model checkpoints
│   │   └── lightning_logs/ # Training logs
│   └── model_metrics_summary.csv  # Model metrics
├── src_v.2/                # Enhanced implementation
│   ├── app*.py             # 17 individual application scripts
│   ├── appv2-1.py          # Main application entry point
│   ├── app_validation.py   # Validation scripts
│   ├── consolidated_model_metrics.csv  # Detailed metrics
│   ├── myplot_app*.png     # Model-specific visualizations
│   └── lightning_logs/     # Extensive training logs (170+ versions)
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## Implemented Models

The project includes implementations of:

- **RNN Variants**
  - Basic RNN
  - LSTM
  - GRU
  - TCN (Temporal Convolutional Network)
  - BiTCN (Bidirectional TCN)
  - Dilated RNN
  - DeepAR

- **Transformer Variants**
  - Vanilla Transformer
  - Informer
  - Autoformer
  - FEDformer
  - PatchTST
  - iTransformer

## Data Sources

- Historical irrigation data from 1975-2015
- Local station data
- Generated forecast data from multiple models
- Model performance metrics

## Usage

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. For original implementation:
```bash
# Training
python src/02_train_model.py

# Prediction
python src/03_predict_test_data.py

# Visualization
python src/app_plot_result_light.py
```

3. For enhanced implementation:
```bash
# Main application
python src_v.2/appv2-1.py

# Specific model execution
python src_v.2/app4.py  # Example for model 4
```

## Results

Model performance metrics are available in:
- `src/model_metrics_summary.csv` (Original)
- `src_v.2/consolidated_model_metrics.csv` (Enhanced)

Visualizations of model forecasts and comparisons are available in the `img/` directory.

## License

[MIT License]
