import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from neuralforecast import NeuralForecast
from neuralforecast.models import VanillaTransformer
from neuralforecast.losses.pytorch import MAE
from neuralforecast.utils import augment_calendar_df
import os


def train_and_predict(Y_df, horizon, output_path, full_horizon, model_name):
    """Treina o modelo VanillaTransformer, salva previsões e métricas nos diretórios especificados."""

    # Nome do modelo aplicado
    applied_model_name = 'VanillaTransformer'

    # Diretórios para salvar o modelo e resultados
    model_output_path = os.path.join(output_path, 'checkpoints', f"{applied_model_name}_{model_name}")
    log_output_path = os.path.join(output_path, 'lightning_logs', f"{applied_model_name}_{model_name}")

    # Criação dos diretórios, caso não existam
    os.makedirs(model_output_path, exist_ok=True)
    os.makedirs(log_output_path, exist_ok=True)

    # Aplicar variáveis de calendário ao Y_df para incluir colunas exógenas diretamente
    Y_df, calendar_cols = augment_calendar_df(df=Y_df, freq='H')
    Y_df['unique_id'] = Y_df['unique_id'].astype(str)  # Garantindo que unique_id seja string
    Y_df['ds'] = pd.to_datetime(Y_df['ds'])  # Garantindo que ds seja datetime
    Y_df = Y_df.sort_values(by=['unique_id', 'ds']).reset_index(drop=True)  # Ordenando por unique_id e ds

    # Treinamento do modelo
    start_time = time.time()
    model = VanillaTransformer(
        h=horizon,
        input_size=24,
        hidden_size=16,
        conv_hidden_size=32,
        n_head=2,
        loss=MAE(),
        scaler_type='robust',
        learning_rate=1e-3,
        max_steps=500
    )
    nf = NeuralForecast(models=[model], freq='H')
    nf.fit(df=Y_df[['unique_id', 'ds', 'y'] + calendar_cols])  # Incluindo variáveis exógenas diretamente
    nf.save(path=model_output_path, model_index=None, overwrite=True, save_dataset=True)
    end_time = time.time()
    print(f'Modelo {model_name} treinado em:', end_time - start_time, 'segundos')

    # Carregando o modelo treinado para previsão
    nf_loaded = NeuralForecast.load(path=model_output_path)

    # Previsões de múltiplos passos
    n_predicts = full_horizon // horizon
    combined_train = Y_df[['unique_id', 'ds', 'y']].copy()  # Usando apenas 'unique_id', 'ds', 'y' inicialmente
    forecasts = []

    for _ in range(n_predicts):
        step_forecast = nf_loaded.predict(df=combined_train)

        # Renomeando a coluna de previsão e garantindo o tipo correto para 'ds'
        step_forecast = step_forecast.rename(columns={applied_model_name: 'y'})
        step_forecast['ds'] = pd.to_datetime(step_forecast['ds'])  # Garantindo que ds é datetime

        # Concatenando previsões para a próxima iteração (sem 'unique_id')
        combined_train = pd.concat([combined_train, step_forecast[['ds', 'y']]], ignore_index=True)
        forecasts.append(step_forecast)

    full_forecast = pd.concat(forecasts, ignore_index=True)
    full_forecast['model_name'] = model_name

    # Salvando previsões em arquivo CSV
    forecast_output_path = os.path.join(model_output_path, f'{applied_model_name}_{model_name}_full_forecast.csv')
    full_forecast.to_csv(forecast_output_path, index=False)

    return full_forecast


if __name__ == '__main__':
    data_path = '../data/data_est_local.csv'
    horizon = 12
    full_horizon = 24
    output_path = './output'

    # Carregar dados e preparar para o treinamento
    Y_df = pd.read_csv(data_path, sep=';', usecols=lambda column: column != 'Unnamed: 19')

    # Ajustar parsing de datas
    Y_df['ds'] = pd.to_datetime(Y_df['Data'] + ' ' + Y_df['Hora'], errors='coerce')
    Y_df = Y_df.dropna(subset=['ds'])
    Y_df['y'] = Y_df['TEMPERATURA DO AR BULBO SECO (°C)']
    Y_df['unique_id'] = 'serie_1'
    Y_df['unique_id'] = Y_df['unique_id'].astype(str)  # Garantindo que unique_id seja string
    Y_df = Y_df.dropna(subset=['y']).sort_values('ds').reset_index(drop=True)

    model_name = 'VanillaTransformer_model'
    forecast = train_and_predict(Y_df, horizon, output_path, full_horizon, model_name)

    # Plot comparativo
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(Y_df['ds'], Y_df['y'], label='Dados Originais', color='black')
    ax1.plot(forecast['ds'], forecast['y'], label=f'Previsão {model_name}', color='blue')
    ax1.set_xlabel('Tempo')
    ax1.set_ylabel('Temperatura (°C)')
    ax1.set_title(f'Previsão do Modelo {model_name}')
    ax1.legend()
    ax1.grid()

    plt.tight_layout()
    plt.show()





