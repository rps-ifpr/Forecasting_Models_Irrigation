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

## Directory Structure Version 1

```plaintext
## Directory Structure Version 2

```plaintext
.
├── .venv/                   # Python virtual environment
├── data/                    # Raw and processed data
├── img/                     # Images generated during model execution
├── src_v2/                  # Updated source code
│   ├── lightning_logs/      # Training and validation logs for the models
│   └── output/              # Scripts and model outputs
│       ├── app4.py          # Updated model 4 script
│       ├── app5.py          # Updated model 5 script
│       ├── app6.py          # Updated model 6 script
│       ├── app7.py          # Updated model 7 script
│       ├── app8.py          # Updated model 8 script
│       ├── app9.py          # Updated model 9 script
│       ├── app10.py         # Updated model 10 script
│       ├── app11.py         # Updated model 11 script
│       ├── app12.py         # Updated model 12 script
│       ├── app13.py         # Updated model 13 script
│       ├── app14.py         # Updated model 14 script
│       ├── app15.py         # Updated model 15 script
│       ├── app16.py         # Updated model 16 script
│       ├── app17.py         # Updated model 17 script
│       ├── app_plot_result_checkpoints.py # Checkpoint visualization script
│       ├── app_plot_result_light.py       # Simplified results visualization script
│       ├── appv2-1.py       # Main script for joint execution
│       ├── consolidated_model_metrics.csv # Consolidated metrics summary
│       ├── myplot_app4.png  # Plot generated for app4
│       ├── myplot_app5.png  # Plot generated for app5
│       ├── myplot_app6.png  # Plot generated for app6
│       ├── myplot_app7.png  # Plot generated for app7
│       ├── myplot_app8.png  # Plot generated for app8
│       ├── myplot_app9.png  # Plot generated for app9
│       ├── myplot_app10.png # Plot generated for app10
│       ├── myplot_app11.png # Plot generated for app11
│       ├── myplot_app12.png # Plot generated for app12
│       └── model_metrics_summary.csv # Metrics summary for models


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
