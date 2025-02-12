import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from neuralforecast.models import Informer
from neuralforecast.losses.pytorch import MAE
from neuralforecast.core import NeuralForecast
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tools.eval_measures import rmse, rmspe, maxabs, meanabs, medianabs
import math
import os


os.environ['NIXTLA_ID_AS_COL'] = '1'


def train_and_predict_autoinformer(Y_df, horizon, config, output_path, full_horizon, model_name):
    """Treina o modelo AutoInformer, salva previsões e métricas nos diretórios especificados."""

    applied_model_name = 'Informer'


    model_output_path = os.path.join(output_path, 'checkpoints', f"{applied_model_name}_{model_name}")
    log_output_path = os.path.join(output_path, 'lightning_logs', f"{applied_model_name}_{model_name}")

    os.makedirs(model_output_path, exist_ok=True)
    os.makedirs(log_output_path, exist_ok=True)


    Y_df['unique_id'] = Y_df['unique_id'].astype(str)
    Y_df['ds'] = pd.to_datetime(Y_df['ds'], errors='coerce')
    Y_df = Y_df.dropna(subset=['ds']).sort_values(by=['unique_id', 'ds']).reset_index(drop=True)


    print(f"Iniciando treinamento do modelo: {model_name} com validação cruzada...")
    start_time = time.time()

    tscv = TimeSeriesSplit(n_splits=3)
    for train_index, test_index in tscv.split(Y_df):
        train_data = Y_df.iloc[train_index]
        test_data = Y_df.iloc[test_index]

        model = Informer(
            h=horizon,
            input_size=config['input_size'],
            hidden_size=config['hidden_size'],
            conv_hidden_size=config['conv_hidden_size'],
            n_head=config['n_head'],
            loss=MAE(),
            scaler_type=config['scaler_type'],
            learning_rate=config['learning_rate'],
            max_steps=config['max_steps']
        )
        nf = NeuralForecast(models=[model], freq='H')
        nf.fit(df=train_data[['unique_id', 'ds', 'y']])


    nf.save(path=model_output_path, model_index=None, overwrite=True, save_dataset=True)
    end_time = time.time()
    print(f"Modelo {model_name} treinado em: {end_time - start_time:.2f} segundos")
    print(f"Modelo {model_name} salvo em: {model_output_path}")


    nf_loaded = NeuralForecast.load(path=model_output_path)
    n_predicts = math.ceil(full_horizon / horizon)
    combined_train = Y_df[['unique_id', 'ds', 'y']].copy()
    forecasts = []

    for _ in range(n_predicts):
        combined_train['unique_id'] = combined_train['unique_id'].astype(str)


        last_date = combined_train['ds'].max()
        future_dates = pd.date_range(last_date, periods=horizon + 1, freq='H')[1:]
        futr_df = pd.DataFrame({
            'unique_id': combined_train['unique_id'].iloc[0],
            'ds': future_dates
        })

        step_forecast = nf_loaded.predict(df=combined_train[['unique_id', 'ds', 'y']], futr_df=futr_df)
        step_forecast = step_forecast.rename(columns={applied_model_name: 'y'})
        combined_train = pd.concat([combined_train, step_forecast[['unique_id', 'ds', 'y']]], ignore_index=True)
        forecasts.append(step_forecast)

    full_forecast = pd.concat(forecasts, ignore_index=True)
    full_forecast['model_name'] = model_name


    forecast_output_path = os.path.join(model_output_path, f'{applied_model_name}_{model_name}_full_forecast.csv')
    full_forecast.to_csv(forecast_output_path, index=False)
    print(f"Previsões salvas em {forecast_output_path}")


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

    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(os.path.join(log_output_path, f'{applied_model_name}_{model_name}_metrics.csv'), index=False)
    print(f"Métricas salvas em {log_output_path}")

    return full_forecast, metrics


if __name__ == '__main__':
    data_path = '../data/data_est_local.csv'
    horizon = 12
    full_horizon = 24
    output_path = './output'


    Y_df = pd.read_csv(data_path, sep=';', usecols=lambda column: column != 'Unnamed: 19')
    Y_df['ds'] = pd.to_datetime(Y_df['Data'] + ' ' + Y_df['Hora'], errors='coerce')
    Y_df = Y_df.dropna(subset=['ds'])
    Y_df['y'] = Y_df['TEMPERATURA DO AR BULBO SECO (°C)']
    Y_df['unique_id'] = 'serie_1'
    Y_df = Y_df.dropna(subset=['y']).sort_values('ds').reset_index(drop=True)


    config = {
        "input_size": horizon,
        "hidden_size": 16,
        "conv_hidden_size": 32,
        "n_head": 2,
        "learning_rate": 1e-3,
        "scaler_type": 'robust',
        "max_steps": 500,
    }

    model_name = 'AutoInformer_model'
    forecast, metrics = train_and_predict_autoinformer(Y_df, horizon, config, output_path, full_horizon, model_name)


    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    ax1.plot(Y_df['ds'], Y_df['y'], label='Dados Originais', color='black')
    ax1.plot(forecast['ds'], forecast['y'], label=f'Previsão {model_name}', color='blue')
    ax1.set_xlabel('Tempo')
    ax1.set_ylabel('Temperatura (°C)')
    ax1.set_title(f'Previsão do Modelo {model_name}')
    ax1.legend()
    ax1.grid()

    metrics_df = pd.DataFrame([metrics])
    metrics_df.set_index('model_name', inplace=True)
    metrics_df.plot(kind='bar', ax=ax2)
    ax2.set_title('Métricas do Modelo')
    ax2.set_xlabel('Modelos')
    ax2.set_ylabel('Valores')
    ax2.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f'{model_name}_forecast_plot.png'))
    plt.show()
