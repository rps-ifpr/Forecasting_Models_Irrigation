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
    """Treina o modelo AutoTFT, salva previsões e Ground Truth para análise detalhada."""

    applied_model_name = 'AutoTFT'
    model_output_path = os.path.join(output_path, 'checkpoints', f"{applied_model_name}_{model_name}")
    log_output_path = os.path.join(output_path, 'lightning_logs', f"{applied_model_name}_{model_name}")

    os.makedirs(model_output_path, exist_ok=True)
    os.makedirs(log_output_path, exist_ok=True)

    print(f"Iniciando treinamento do modelo: {model_name}...")
    start_time = time.time()
    models = [AutoTFT(h=horizon, loss=MAE(), config=config, num_samples=10)]
    nf = NeuralForecast(models=models, freq='H')

    nf.fit(df=Y_df[['unique_id', 'ds', 'y']])
    nf.save(path=model_output_path, model_index=None, overwrite=True, save_dataset=True)
    end_time = time.time()
    print(f"Modelo {model_name} treinado em:", end_time - start_time, 'segundos')

    # Previsões
    nf_loaded = NeuralForecast.load(path=model_output_path)
    n_predicts = math.ceil(full_horizon / horizon)
    combined_train = Y_df[['unique_id', 'ds', 'y']].copy()
    forecasts = []

    for _ in range(n_predicts):
        step_forecast = nf_loaded.predict(df=combined_train)
        step_forecast = step_forecast.rename(columns={applied_model_name: 'y_pred'})
        step_forecast = step_forecast.reset_index()
        combined_train = pd.concat([combined_train, step_forecast[['unique_id', 'ds', 'y_pred']]], ignore_index=True)
        forecasts.append(step_forecast)

    full_forecast = pd.concat(forecasts, ignore_index=True)
    full_forecast = full_forecast.merge(Y_df[['ds', 'y']], on='ds', how='left')  # Adicionar Ground Truth
    full_forecast['model_name'] = model_name

    # Salvando previsões e Ground Truth
    forecast_output_path = os.path.join(model_output_path, f'{applied_model_name}_{model_name}_ground_truth.csv')
    full_forecast.to_csv(forecast_output_path, index=False)

    # Calculando métricas
    y_pred = full_forecast['y_pred'].dropna().values
    y_true = full_forecast['y'].dropna().values
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

    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(os.path.join(log_output_path, f'{applied_model_name}_{model_name}_metrics.csv'), index=False)

    return full_forecast, metrics


if __name__ == '__main__':
    data_path = '../data/data_est_local.csv'
    horizon = 1
    full_horizon = 20
    output_path = './output'

    # Carregar dados
    Y_df = pd.read_csv(data_path, sep=';', usecols=lambda column: column != 'Unnamed: 19')
    Y_df['ds'] = pd.to_datetime(Y_df['Data'] + ' ' + Y_df['Hora'], errors='coerce')
    Y_df = Y_df.dropna(subset=['ds'])

    Y_df['y'] = Y_df['TEMPERATURA DO AR BULBO SECO (°C)']
    Y_df['unique_id'] = 'serie_1'
    Y_df = Y_df.dropna(subset=['y']).sort_values('ds').reset_index(drop=True)

    print("Intervalo de Datas:", Y_df['ds'].min(), "até", Y_df['ds'].max())

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
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(Y_df['ds'], Y_df['y'], label='Ground Truth (Dados Originais)', color='black')
    ax.plot(forecast['ds'], forecast['y_pred'], label=f'Previsão {model_name}', color='blue')
    ax.set_xlabel('Tempo')
    ax.set_ylabel('Temperatura (°C)')
    ax.set_title(f'Previsão x Ground Truth - {model_name}')
    ax.legend()
    ax.grid()

    plt.tight_layout()
    plt.show()

