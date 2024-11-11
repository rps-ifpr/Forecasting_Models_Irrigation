import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from neuralforecast.models import iTransformer
from neuralforecast.losses.pytorch import MSE, MAE
from neuralforecast.core import NeuralForecast
from statsmodels.tools.eval_measures import rmse, rmspe, maxabs, meanabs, medianabs
import math
import os

# Definir a variável de ambiente para suprimir o aviso de `FutureWarning`
os.environ['NIXTLA_ID_AS_COL'] = '1'

def train_and_predict_itransformer(Y_df, horizon, config, output_path, full_horizon, model_name):
    """Treina o modelo AutoiTransformer e salva previsões e métricas nos diretórios especificados."""
    applied_model_name = 'iTransformer'

    # Diretórios para salvar o modelo e resultados
    model_output_path = os.path.join(output_path, 'checkpoints', f"{applied_model_name}_{model_name}")
    log_output_path = os.path.join(output_path, 'lightning_logs', f"{applied_model_name}_{model_name}")

    os.makedirs(model_output_path, exist_ok=True)
    os.makedirs(log_output_path, exist_ok=True)

    # Ajustando DataFrame de entrada
    Y_df['unique_id'] = Y_df['unique_id'].astype(str)
    Y_df['ds'] = pd.to_datetime(Y_df['ds'], errors='coerce')
    Y_df = Y_df.dropna(subset=['ds']).sort_values(by=['unique_id', 'ds']).reset_index(drop=True)

    # Treinamento do modelo
    start_time = time.time()
    model = iTransformer(
        h=horizon,
        input_size=config['input_size'],
        n_series=config['n_series'],
        hidden_size=config['hidden_size'],
        n_heads=config['n_heads'],
        e_layers=config['e_layers'],
        d_layers=config['d_layers'],
        d_ff=config['d_ff'],
        factor=config['factor'],
        dropout=config['dropout'],
        use_norm=config['use_norm'],
        loss=MSE(),
        valid_loss=MAE(),
        early_stop_patience_steps=config['early_stop_patience_steps'],
        batch_size=config['batch_size']
    )
    nf = NeuralForecast(models=[model], freq='H')
    nf.fit(df=Y_df[['unique_id', 'ds', 'y']], val_size=horizon)
    nf.save(path=model_output_path, model_index=None, overwrite=True, save_dataset=True)
    end_time = time.time()

    # Carregando o modelo treinado para previsão
    nf_loaded = NeuralForecast.load(path=model_output_path)

    # Previsões de múltiplos passos
    n_predicts = math.ceil(full_horizon / horizon)
    combined_train = Y_df[['unique_id', 'ds', 'y']].copy()
    forecasts = []

    for _ in range(n_predicts):
        combined_train['unique_id'] = combined_train['unique_id'].astype(str)
        last_date = combined_train['ds'].max()
        future_dates = pd.date_range(last_date, periods=horizon + 1, freq='H')[1:]
        futr_df = pd.DataFrame({'unique_id': combined_train['unique_id'].iloc[0], 'ds': future_dates})
        step_forecast = nf_loaded.predict(df=combined_train[['unique_id', 'ds', 'y']], futr_df=futr_df)
        step_forecast = step_forecast.rename(columns={applied_model_name: 'y'})
        step_forecast['ds'] = pd.to_datetime(step_forecast['ds'], errors='coerce')
        combined_train = pd.concat([combined_train, step_forecast[['unique_id', 'ds', 'y']]], ignore_index=True)
        forecasts.append(step_forecast)

    full_forecast = pd.concat(forecasts, ignore_index=True)
    full_forecast['model_name'] = model_name

    # Salvando previsões
    forecast_output_path = os.path.join(model_output_path, f'{applied_model_name}_{model_name}_full_forecast.csv')
    full_forecast.to_csv(forecast_output_path, index=False)

    # Calculando métricas
    y_pred = full_forecast['y'].values
    y_true = Y_df['y'].iloc[-len(y_pred):].values
    metrics = {
        'RMSE': rmse(y_true, y_pred),
        'RMSPE': rmspe(y_true, y_pred),
        'Max Abs Error': maxabs(y_true, y_pred),
        'Mean Abs Error': meanabs(y_true, y_pred),
        'Median Abs Error': medianabs(y_true, y_pred)
    }

    # Salvando métricas
    metrics_df = pd.DataFrame([metrics])
    metrics_output_path = os.path.join(log_output_path, f'{applied_model_name}_{model_name}_metrics.csv')
    metrics_df.to_csv(metrics_output_path, index=False)

    return full_forecast, metrics

if __name__ == '__main__':
    data_path = '../data/data_est_local.csv'
    horizon = 12
    full_horizon = 24
    output_path = './output'

    # Carregar dados e preparar para o treinamento
    Y_df = pd.read_csv(data_path, sep=';', usecols=lambda column: column != 'Unnamed: 19')
    Y_df['ds'] = pd.to_datetime(Y_df['Data'] + ' ' + Y_df['Hora'], errors='coerce')
    Y_df = Y_df.dropna(subset=['ds'])
    Y_df['y'] = Y_df['TEMPERATURA DO AR BULBO SECO (°C)']
    Y_df['unique_id'] = 'serie_1'
    Y_df = Y_df.dropna(subset=['y']).sort_values('ds').reset_index(drop=True)

    # Configuração do modelo
    config = {
        "input_size": 24,
        "n_series": 1,
        "hidden_size": 128,
        "n_heads": 4,
        "e_layers": 3,
        "d_layers": 2,
        "d_ff": 256,
        "factor": 2,
        "dropout": 0.1,
        "use_norm": True,
        "early_stop_patience_steps": 3,
        "batch_size": 64
    }

    model_name = 'AutoiTransformer_model'
    forecast, metrics = train_and_predict_itransformer(Y_df, horizon, config, output_path, full_horizon, model_name)

    # Visualização
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    ax1.plot(Y_df['ds'], Y_df['y'], label='Dados Originais', color='black')
    ax1.plot(forecast['ds'], forecast['y'], label='Previsões', color='blue')
    ax1.legend()
    ax1.set_title('Previsão vs Dados Originais')
    ax1.set_xlabel('Data')
    ax1.set_ylabel('Valor')

    ax2.bar(metrics.keys(), [float(v) for v in metrics.values()])
    ax2.set_title('Métricas do Modelo')
    ax2.set_xlabel('Métrica')
    ax2.set_ylabel('Valor')

    plt.tight_layout()
    plt.show()

