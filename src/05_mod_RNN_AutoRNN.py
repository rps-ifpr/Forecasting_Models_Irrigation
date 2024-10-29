import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from neuralforecast import NeuralForecast
from neuralforecast.models import RNN
from neuralforecast.losses.pytorch import MQLoss
import os
import math


def train_and_predict_auto_rnn(Y_df, horizon, config, output_path, full_horizon, model_name):
    """Treina o modelo AutoRNN, salva previsões e métricas nos diretórios especificados."""

    # Nome do modelo aplicado
    applied_model_name = 'AutoRNN'

    # Diretórios para salvar o modelo e resultados
    model_output_path = os.path.join(output_path, 'checkpoints', f"{applied_model_name}_{model_name}")
    log_output_path = os.path.join(output_path, 'lightning_logs', f"{applied_model_name}_{model_name}")

    # Criação dos diretórios, caso não existam
    os.makedirs(model_output_path, exist_ok=True)
    os.makedirs(log_output_path, exist_ok=True)

    # Treinamento do modelo
    start_time = time.time()
    models = [RNN(
        h=horizon,
        input_size=-1,
        inference_input_size=24,
        loss=MQLoss(level=[80, 90]),
        scaler_type='robust',
        encoder_n_layers=config["encoder_n_layers"],
        encoder_hidden_size=config["encoder_hidden_size"],
        context_size=config["context_size"],
        decoder_hidden_size=config["decoder_hidden_size"],
        decoder_layers=config["decoder_layers"],
        max_steps=config["max_steps"]
    )]
    nf = NeuralForecast(models=models, freq='M')
    nf.fit(df=Y_df[['unique_id', 'ds', 'y']])
    nf.save(path=model_output_path, model_index=None, overwrite=True, save_dataset=True)
    end_time = time.time()
    print(f'Modelo {model_name} treinado em:', end_time - start_time, 'segundos')

    # Carregando o modelo treinado para previsão
    nf_loaded = NeuralForecast.load(path=model_output_path)

    # Previsões de múltiplos passos
    n_predicts = math.ceil(full_horizon / horizon)
    combined_train = Y_df[['unique_id', 'ds', 'y']].copy()
    forecasts = []

    for _ in range(n_predicts):
        step_forecast = nf_loaded.predict(df=combined_train)

        # Verificar as colunas do dataframe de previsão para identificar o nome da coluna de previsões
        print("Colunas de step_forecast:", step_forecast.columns)

        # Renomear a coluna de previsão para 'y' (assumindo que a coluna de previsão seja identificada pelo nome do modelo)
        forecast_column = step_forecast.columns[-1]  # A coluna de previsão é provavelmente a última
        step_forecast = step_forecast.rename(columns={forecast_column: 'y'})

        step_forecast = step_forecast.reset_index()
        step_forecast['unique_id'] = step_forecast.get('unique_id', 'serie_1').astype(str)
        combined_train = pd.concat([combined_train, step_forecast[['unique_id', 'ds', 'y']]], ignore_index=True)
        forecasts.append(step_forecast)

    full_forecast = pd.concat(forecasts, ignore_index=True)
    full_forecast['model_name'] = model_name

    # Salvando previsões em arquivo CSV
    forecast_output_path = os.path.join(model_output_path, f'{applied_model_name}_{model_name}_full_forecast.csv')
    full_forecast.to_csv(forecast_output_path, index=False)

    return full_forecast


if __name__ == '__main__':
    data_path = '../data/data_est_local.csv'
    horizon = 1
    full_horizon = 20
    output_path = './output'

    # Carregar dados e preparar para o treinamento
    Y_df = pd.read_csv(data_path, sep=';', usecols=lambda column: column != 'Unnamed: 19')

    # Ajustar parsing de datas
    Y_df['ds'] = pd.to_datetime(Y_df['Data'] + ' ' + Y_df['Hora'], errors='coerce')
    Y_df = Y_df.dropna(subset=['ds'])
    Y_df['y'] = Y_df['TEMPERATURA DO AR BULBO SECO (°C)']
    Y_df['unique_id'] = 'serie_1'
    Y_df = Y_df.dropna(subset=['y']).sort_values('ds').reset_index(drop=True)

    # Configuração do modelo AutoRNN
    config = {
        "encoder_n_layers": 2,
        "encoder_hidden_size": 128,
        "context_size": 10,
        "decoder_hidden_size": 128,
        "decoder_layers": 2,
        "max_steps": 300
    }

    model_name = 'AutoRNN_model'
    forecast = train_and_predict_auto_rnn(Y_df, horizon, config, output_path, full_horizon, model_name)

    # Plot comparativo
    plt.plot(Y_df['ds'], Y_df['y'], label='Dados Originais', color='black')
    plt.plot(forecast['ds'], forecast['y'], label=f'Previsão {model_name}', color='blue')
    plt.xlabel('Tempo')
    plt.ylabel('Temperatura (°C)')
    plt.title(f'Previsão do Modelo {model_name}')
    plt.legend()
    plt.grid(linestyle='-', which='both')
    plt.show()


