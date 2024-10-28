import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from neuralforecast.auto import AutoTFT, Autoformer
from neuralforecast.losses.pytorch import MAE
from ray import tune
from neuralforecast.core import NeuralForecast
from statsmodels.tools.eval_measures import rmse, rmspe, maxabs, meanabs, medianabs
import math
import os

def train_and_predict(Y_df, horizon, config, output_path, full_horizon, model_name):
    """Treina o modelo, salva e faz previsões, retorna previsões e métricas."""

    applied_model_name = model_name

    model_output_path = os.path.join(output_path, f"{applied_model_name}_{model_name}")
    if not os.path.exists(model_output_path):
        os.makedirs(model_output_path)

    start_time = time.time()
    if model_name == 'AutoTFT':
        models = [AutoTFT(h=horizon, loss=MAE(), config=config, num_samples=10)]
    elif model_name == 'Autoformer':
        # Adicionando decoder_input_size_multiplier com um valor adequado
        config["decoder_input_size_multiplier"] = 0.5
        models = [Autoformer(h=horizon, input_size=horizon, loss=MAE(), config=config, num_samples=10)]

    # Adicione outros modelos conforme necessário

    nf = NeuralForecast(models=models, freq='H')
    nf.fit(df=Y_df[['unique_id', 'ds', 'y']])
    nf.save(path=model_output_path, model_index=None, overwrite=True, save_dataset=True)
    end_time = time.time()
    print(f'Modelo {model_name} treinado em:', end_time - start_time, 'segundos')

    nf_loaded = NeuralForecast.load(path=model_output_path)

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

    forecast_output_path = os.path.join(model_output_path, f'{applied_model_name}_{model_name}_full_forecast.csv')
    full_forecast.to_csv(forecast_output_path, index=False)

    y_pred = full_forecast['y'].values
    y_true = Y_df['y'].iloc[-len(y_pred):].values
    rmse_value = rmse(y_true, y_pred)
    rmspe_value = rmspe(y_true, y_pred)
    maxabs_value = maxabs(y_true, y_pred)
    meanabs_value = meanabs(y_true, y_pred)
    medianabs_value = medianabs(y_true, y_pred)

    return full_forecast, {
        'model_name': model_name,
        'RMSE': rmse_value,
        'RMSPE': rmspe_value,
        'Max Abs Error': maxabs_value,
        'Mean Abs Error': meanabs_value,
        'Median Abs Error': medianabs_value
    }


if __name__ == '__main__':
    data_path = '../data/data_est_local.csv'
    horizon = 1
    full_horizon = 20
    output_path = 'checkpoints/'

    Y_df = pd.read_csv(data_path, sep=';')
    Y_df['ds'] = pd.to_datetime(Y_df['Data'] + ' ' + Y_df['Hora'], format='%Y-%m-%d %H:%M:%S')
    Y_df['y'] = Y_df['TEMPERATURA DO AR BULBO SECO (°C)']
    Y_df['unique_id'] = 'serie_1'
    Y_df = Y_df.dropna(subset=['y']).sort_values('ds').reset_index(drop=True)

    models_to_test = [
        {
            'model_name': 'AutoTFT',
            'config': {
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
        },
        {
            'model_name': 'Autoformer',
            'config': {
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
        }
    ]

    all_forecasts = []
    all_metrics = []

    for model_info in models_to_test:
        model_name = model_info['model_name']
        config = model_info['config']
        print(f'Treinando e avaliando o modelo {model_name}')
        forecast, metrics = train_and_predict(Y_df, horizon, config, output_path, full_horizon, model_name)
        all_forecasts.append(forecast)
        all_metrics.append(metrics)

    combined_forecasts = pd.concat(all_forecasts, ignore_index=True)
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.set_index('model_name', inplace=True)

    # Plot comparativo
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

    # Previsões dos modelos
    ax1.plot(Y_df['ds'], Y_df['y'], label='Dados Originais', color='black')
    for model_name in combined_forecasts['model_name'].unique():
        model_forecast = combined_forecasts[combined_forecasts['model_name'] == model_name]
        ax1.plot(model_forecast['ds'], model_forecast['y'], label=f'Previsão {model_name}')
    ax1.set_xlabel('Tempo')
    ax1.set_ylabel('Temperatura (°C)')
    ax1.set_title('Comparação das Previsões dos Modelos')
    ax1.legend()
    ax1.grid(linestyle='-', which='both')

    # Métricas dos modelos
    metrics_df.plot(kind='bar', ax=ax2)
    ax2.set_title('Comparação das Métricas dos Modelos')
    ax2.set_xlabel('Modelos')
    ax2.set_ylabel('Valores')
    ax2.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()
