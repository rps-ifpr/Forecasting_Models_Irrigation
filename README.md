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
.
├── .venv/                   # Ambiente virtual Python
├── data/                    # Dados brutos e processados
├── img/                     # Imagens geradas durante a execução dos modelos
├── src/                     # Código-fonte principal
│   └── output/              # Scripts e saídas dos modelos
│       ├── 01_model.py                      # Script base do modelo inicial
│       ├── 02_train_model.py                # Script de treinamento do modelo
│       ├── 03_predict_test_data.py          # Previsão com dados de teste
│       ├── 04_mod_Transfor_TFT.py           # Modelo Temporal Fusion Transformer (TFT)
│       ├── 05_mod_RNN_AutoRNN.py            # Modelo RNN com automação
│       ├── 06_mod_RNN_AutoLSTM.py           # Modelo LSTM com automação
│       ├── 07_mod_RNN_AutoGRU.py            # Modelo GRU com automação
│       ├── 08_mod_RNN_AutoTCN.py            # Modelo TCN (Temporal Convolutional Network)
│       ├── 09_mod_RNN_DeepAR.py             # Modelo DeepAR
│       ├── 10_mod_RNN_AutoDilatedRNN.py     # Modelo RNN com dilatação
│       ├── 11_mod_RNN_AutoBiTCN.py          # Modelo BiTCN
│       ├── 12_mod_Transfor_AutoVanilla.py   # Modelo Transformer Vanilla
│       ├── 13_mod_Transfor_AutoInformer.py  # Modelo Informer
│       ├── 14_mod_Transfor_Autoformer.py    # Modelo Autoformer
│       ├── 15_mod_Transfor_AutoFEDformer.py # Modelo FEDformer
│       ├── 16_mod_Transfor_AutoPatchTST.py  # Modelo PatchTST
│       ├── 17_mod_Transfor_AutoTransformer.py # Modelo Transformer genérico
│       ├── app_plot_result_checkpoints.py   # Script para visualização de checkpoints
│       ├── app_plot_result_light.py         # Script para visualização de resultados simplificados
│       └── model_metrics_summary.csv        # Resumo das métricas dos modelos

This is the second version of the project, with significant improvements over the first. The main difference is the full implementation of cross-validation for all models, along with other structural changes.

## Directory Structure Version 2

```plaintext
.
├── src_v2/                       # Código-fonte atualizado
│   ├── lightning_logs/           # Logs de treinamento e validação dos modelos
│   └── output/                   # Scripts e saídas dos modelos
│       ├── app4.py               # Modelo 4 atualizado
│       ├── app5.py               # Modelo 5 atualizado
│       ├── app6.py               # Modelo 6 atualizado
│       ├── app7.py               # Modelo 7 atualizado
│       ├── app8.py               # Modelo 8 atualizado
│       ├── app9.py               # Modelo 9 atualizado
│       ├── app10.py              # Modelo 10 atualizado
│       ├── app11.py              # Modelo 11 atualizado
│       ├── app12.py              # Modelo 12 atualizado
│       ├── app13.py              # Modelo 13 atualizado
│       ├── app14.py              # Modelo 14 atualizado
│       ├── app15.py              # Modelo 15 atualizado
│       ├── app16.py              # Modelo 16 atualizado
│       ├── app17.py              # Modelo 17 atualizado
│       ├── app_plot_result_checkpoints.py # Visualização dos checkpoints
│       ├── app_plot_result_light.py       # Visualização simplificada dos resultados
│       ├── appv2-1.py             # Script principal para execução conjunta
│       └── consolidated_model_metrics.csv # Resumo consolidado das métricas
│       ├── myplot_app4.png        # Gráfico gerado para app4
│       ├── myplot_app5.png        # Gráfico gerado para app5
│       ├── myplot_app6.png        # Gráfico gerado para app6
│       ├── myplot_app7.png        # Gráfico gerado para app7
│       ├── myplot_app8.png        # Gráfico gerado para app8
│       ├── myplot_app9.png        # Gráfico gerado para app9
│       ├── myplot_app10.png       # Gráfico gerado para app10
│       ├── myplot_app11.png       # Gráfico gerado para app11
│       ├── myplot_app12.png       # Gráfico gerado para app12

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
