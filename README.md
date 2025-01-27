# Transformer Models and Recurrent Neural Networks Applied to Meteorological Data with NeuralForecast
![](https://i.imgur.com/jYDN7PL.png)

<p align="center">
  <img alt="GitHub language count" src="https://img.shields.io/github/languages/count/cubos-academy/academy-template-readme-projects?color=%2304D361">

  <img alt="Repository size" src="https://img.shields.io/github/repo-size/cubos-academy/academy-template-readme-projects">
  
  <a href="https://github.com/cubos-academy/academy-template-readme-projects/commits/main">
    <img alt="GitHub last commit" src="https://img.shields.io/github/last-commit/cubos-academy/academy-template-readme-projects">
  </a>
  
  <!-- <img alt="License" src="https://img.shields.io/badge/license-MIT-brightgreen"> -->
  
   <a href="https://github.com/cubos-academy/academy-template-readme-projects/stargazers">
    <img alt="Stargazers" src="https://img.shields.io/github/stars/cubos-academy/academy-template-readme-projects?style=social">
  </a>
  
<h4 align="center"> 
	🚧 Forecasting_Models_Irrigation 🚧
</h4>

<p align="center">
	<!--<img alt="Status In Development" src="https://img.shields.io/badge/STATUS-IN%20DEVELOPMENT-green"> -->
	<img alt="Status Completed" src="https://img.shields.io/badge/STATUS-COMPLETED-brightgreen">  
</p>

<p align="center">
 <a href="#-about-the-project">About</a> •
 <a href="#-features">Features</a> •
 <a href="#-layout">Layout</a> • 
 <a href="#-how-to-run-the-project">How to Run</a> • 
 <a href="#-technologies">Technologies</a> • 
 <a href="#-contributors">Contributors</a> • 
 <a href="#-author">Author</a> • 
 <a href="#user-content--license">License</a>
</p>

## 💻 About the project

With the increasing demand for accurate weather forecasts, this project explores the use of neural networks, especially Transformer models and RNNs, to capture complex patterns in meteorological time series. Fourteen models were evaluated using the [NeuralForecast](https://github.com/Nixtla/neuralforecast) library, employing metrics such as RMSE, RMSPE, Max Abs Error, Mean Abs Error, and Median Abs Error.

The project structure is organized as described below.


## Directory Structure

```
.
├── data/                   # Historical and forecast data
│   ├── dados1975-2015.csv  # Main historical dataset (1975-2015)
│   ├── data_est_local.CSV  # Local station data - dataset
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

## ⚙️ Features
- Evaluation of 14 neural network models applied to meteorological time series forecasting:
  - **Transformer-based Models:**
    - Informer
    - Former
    - PatchTST
    - FEDformer
    - VanillaTransformer
    - iTransformer
  - **RNN-based Models:**
    - RNN
    - LSTM
    - GRU
    - TCN
    - DeepAR
    - DilatedRNN
    - BiTCN

- Use of real meteorological data collected from local stations, with hourly measurements over a one-year period (2023).
- Comparison of model performance using multiple error metrics:
  - RMSE (Root Mean Squared Error)
  - RMSPE (Root Mean Squared Percentage Error)
  - Max Abs Error
  - Mean Abs Error
  - Median Abs Error

- Automatic configuration and hyperparameter tuning:
  - Cross-validation and early stopping techniques.
  - Support for exogenous variables to improve predictive accuracy.

- Implementation of short, medium, and long-term forecasts:
  - Recursive forecasting (using previous predictions as inputs).
  - Direct forecasting (generating all horizon steps at once).

- Visualization and analysis of results:
  - Comparative performance charts across metrics for the models.
  - Detailed analysis of forecasts and seasonal patterns captured by the models.

- Storage and organization of results:
  - Trained models and validation logs stored in the `checkpoints` and `lightning_logs` directories.
  - Code and results made available in the GitHub repository for transparency and reproducibility.

## 🛣️ How to Run the Project

### Prerequisites

Before starting, ensure you have installed:
- [Python 3.13](https://www.python.org/downloads/)
- Specific libraries: Check the `requirements.txt` file in the repository.

### Execution

1. Clone this repository:
   ```bash
   git clone https://github.com/rps-ifpr/Forecasting_Models_Irrigation.git

## 🛠 Technologies

The project was developed using the following technologies and tools:

- **Language:** Python 3.13
- **Libraries:**
  - **[NeuralForecast](https://github.com/Nixtla/neuralforecast):** Advanced tool for time series forecasting with support for neural network-based models (Transformers and RNNs).
  - **[PyTorch](https://pytorch.org/):** Machine learning framework used to implement and train the models.
  - **[Pandas](https://pandas.pydata.org/):** Library for data manipulation and analysis.
  - **[Matplotlib](https://matplotlib.org/):** Library for data visualization and creation of comparative charts.
- **Additional Features:**
  - **Automatic Hyperparameter Tuning:** Using techniques like `early stopping` and cross-validation.
  - **Support for Exogenous Variables:** Integration of external data to enhance forecasts.

## 🧑‍💻 Author

This project was developed by **Rogério Pereira dos Santos**, a researcher and developer focused on neural networks applied to meteorological time series forecasting.

- **Institution:** Federal Institute of Paraná (IFPR)
- **Contact:**
  - [LinkedIn](https://www.linkedin.com/in/rogerio-dosantos) — Connect to discuss neural networks and climate forecasting projects.
  - [Email](mailto:rogerio.dosantos@ifpr.edu.br) — For questions or collaborations related to the project.
- **Publications and Contributions:**
  - Academic publications in climate forecasting and machine learning.
  - Experience with technologies applied to precision agriculture and sustainability.

Feel free to get in touch or explore other projects on [GitHub](https://github.com/rps-ifpr).

## 💪 How to Contribute to the Project
1. **Fork** the project.
2. Create a new branch for your changes: `git checkout -b my-feature`
3. Save your changes and create a commit message describing what you did: `git commit -m "feature: My new feature"`
4. Push your changes: `git push origin my-feature`
> If you have any questions, check this [guide on how to contribute on GitHub](./CONTRIBUTING.md).

## 📝 License
<!-- This project is under the [MIT](./LICENSE) license. -->
