import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from neuralforecast.auto import AutoTFT
from neuralforecast.losses.pytorch import MAE
from ray import tune
from neuralforecast.core import NeuralForecast
from statsmodels.tools.eval_measures import rmse, rmspe, maxabs, meanabs, medianabs
import math
import os

def train_and_predict(Y_df, horizon, config, output_path, full_horizon, model_name):
    """Treina o modelo AutoTFT, salva previsões e métricas nos diretórios especificados."""

    # Nome do modelo aplicado
    applied_model_name = 'AutoTFT'

    # Diretórios para salvar o modelo e resultados
    model_output_path = os.path.join(output_path, 'checkpoints', f"{applied_model_name}_{model_name}")
    log_output_path = os.path.join(output_path, 'lightning_logs', f"{applied_model_name}_{model_name}")

    # Criação dos diretórios, caso não existam
    os.makedirs(model_output_path, exist_ok=True)
    os.makedirs(log_output_path, exist_ok=True)

    # Treinamento do modelo
    start_time = time.time()
    models = [AutoTFT(h=horizon, loss=MAE(), config=config, num_samples=10)]
    nf = NeuralForecast(models=models, freq='H')
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
        step_forecast = step_forecast.rename(columns={applied_model_name: 'y'})
        step_forecast = step_forecast.reset_index()
        step_forecast['unique_id'] = step_forecast.get('unique_id', 'serie_1').astype(str)
        combined_train = pd.concat([combined_train, step_forecast[['unique_id', 'ds', 'y']]], ignore_index=True)
        forecasts.append(step_forecast)

    full_forecast = pd.concat(forecasts, ignore_index=True)
    full_forecast['model_name'] = model_name

    # Salvando previsões em arquivo CSV
    forecast_output_path = os.path.join(model_output_path, f'{applied_model_name}_{model_name}_full_forecast.csv')
    full_forecast.to_csv(forecast_output_path, index=False)

    # Calculando métricas
    y_pred = full_forecast['y'].values
    y_true = Y_df['y'].iloc[-len(y_pred):].values
    rmse_value = rmse(y_true, y_pred)
    rmspe_value = rmspe(y_true, y_pred)
    maxabs_value = maxabs(y_true, y_pred)
    meanabs_value = meanabs(y_true, y_pred)
    medianabs_value = medianabs(y_true, y_pred)

    metrics = {
        'model_name': model_name,
        'RMSE': rmse_value,
        'RMSPE': rmspe_value,
        'Max Abs Error': maxabs_value,
        'Mean Abs Error': meanabs_value,
        'Median Abs Error': medianabs_value
    }

    # Salvando métricas no diretório lightning_logs
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(os.path.join(log_output_path, f'{applied_model_name}_{model_name}_metrics.csv'), index=False)

    return full_forecast, metrics


if __name__ == '__main__':
    data_path = '../data/data_est_local.csv'
    horizon = 1
    full_horizon = 20
    output_path = './output'

    # Carregar dados e preparar para o treinamento
    Y_df = pd.read_csv(data_path, sep=';', usecols=lambda column: column != 'Unnamed: 19')

    # Ajustar parsing de datas
    Y_df['ds'] = pd.to_datetime(Y_df['Data'] + ' ' + Y_df['Hora'], errors='coerce')

    # Remover linhas com datas não parseáveis
    Y_df = Y_df.dropna(subset=['ds'])

    # Preparar variável alvo
    Y_df['y'] = Y_df['TEMPERATURA DO AR BULBO SECO (°C)']

    # Atribuir ID único
    Y_df['unique_id'] = 'serie_1'

    # Remover linhas com valores alvo ausentes e ordenar
    Y_df = Y_df.dropna(subset=['y']).sort_values('ds').reset_index(drop=True)

    # Verificar dados
    print("Intervalo de Datas:", Y_df['ds'].min(), "até", Y_df['ds'].max())
    print("Total de Linhas:", Y_df.shape[0])

    # Verificar se os dados cobrem o intervalo esperado
    expected_start_date = pd.to_datetime('2023-01-01 00:00:00')
    expected_end_date = pd.to_datetime('2023-12-31 00:00:00')

    if Y_df['ds'].min() > expected_start_date or Y_df['ds'].max() < expected_end_date:
        print("Aviso: Os dados não cobrem todo o intervalo de datas esperado.")
        # Você pode optar por encerrar ou lidar com este caso conforme necessário

    # Configuração do modelo AutoTFT
    config = {
        "input_size": tune.choice([horizon]),
        "hidden_size": tune.choice([8, 32]),
        "n_head": tune.choice([2, 8]),
        "learning_rate": tune.loguniform(1e-4, 1e-1),
        "scaler_type": tune.choice(['robust', 'standard']),
        "max_steps": tune.choice([500, 1000]),
        "windows_batch_size": tune.choice([8, 32]),
        "check_val_every_n_epoch": tune.choice([100]),
        "random_seed": tune.randint(1, 20),
    }

    model_name = 'AutoTFT_model'
    forecast, metrics = train_and_predict(Y_df, horizon, config, output_path, full_horizon, model_name)

    # Plot comparativo
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

    # Previsões do modelo AutoTFT
    ax1.plot(Y_df['ds'], Y_df['y'], label='Dados Originais', color='black')
    ax1.plot(forecast['ds'], forecast['y'], label=f'Previsão {model_name}', color='blue')
    ax1.set_xlabel('Tempo')
    ax1.set_ylabel('Temperatura (°C)')
    ax1.set_title(f'Previsão do Modelo {model_name}')
    ax1.legend()
    ax1.grid(linestyle='-', which='both')

    # Métricas do modelo
    metrics_df = pd.DataFrame([metrics])
    metrics_df.set_index('model_name', inplace=True)
    metrics_df.plot(kind='bar', ax=ax2)
    ax2.set_title('Métricas do Modelo')
    ax2.set_xlabel('Modelos')
    ax2.set_ylabel('Valores')
    ax2.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show() 