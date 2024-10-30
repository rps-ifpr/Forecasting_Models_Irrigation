import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from neuralforecast import NeuralForecast
from neuralforecast.models import VanillaTransformer
from neuralforecast.losses.pytorch import MAE
from neuralforecast.utils import augment_calendar_df
from statsmodels.tools.eval_measures import rmse, rmspe, maxabs, meanabs, medianabs
import math
import os


def train_and_predict_auto_vanilla(Y_df, horizon, config, output_path, full_horizon, model_name):
    """Treina o modelo AutoVanilla, salva previsões e métricas nos diretórios especificados."""

    # Nome do modelo aplicado
    applied_model_name = 'AutoVanilla'

    # Diretórios para salvar o modelo e resultados
    model_output_path = os.path.join(output_path, 'checkpoints', f"{applied_model_name}_{model_name}")
    log_output_path = os.path.join(output_path, 'lightning_logs', f"{applied_model_name}_{model_name}")

    # Criação dos diretórios, caso não existam
    os.makedirs(model_output_path, exist_ok=True)
    os.makedirs(log_output_path, exist_ok=True)

    # Adicionar variáveis de calendário ao dataset
    Y_df_augmented, calendar_cols = augment_calendar_df(df=Y_df, freq='H')

    # Treinamento do modelo
    start_time = time.time()
    models = [VanillaTransformer(
        h=horizon,
        input_size=config["input_size"],
        hidden_size=config["hidden_size"],
        conv_hidden_size=config["conv_hidden_size"],
        n_head=config["n_head"],
        loss=MAE(),
        futr_exog_list=calendar_cols,
        scaler_type=config["scaler_type"],
        learning_rate=config["learning_rate"],
        max_steps=config["max_steps"]
    )]
    nf = NeuralForecast(models=models, freq='H')
    nf.fit(df=Y_df_augmented[['unique_id', 'ds', 'y'] + calendar_cols])
    nf.save(path=model_output_path, model_index=None, overwrite=True, save_dataset=True)
    end_time = time.time()
    print(f'Modelo {model_name} treinado em:', end_time - start_time, 'segundos')

    # Carregando o modelo treinado para previsão
    nf_loaded = NeuralForecast.load(path=model_output_path)

    # Define `combined_train` como uma cópia do conjunto de treino
    combined_train = Y_df_augmented[['unique_id', 'ds', 'y'] + calendar_cols].copy()

    # Gera o dataframe futuro esperado
    futr_df = nf_loaded.make_future_dataframe(df=combined_train, h=full_horizon)

    # Adiciona as colunas de calendário no futr_df gerado
    futr_df, _ = augment_calendar_df(futr_df, freq='H')

    # Previsões de múltiplos passos
    n_predicts = math.ceil(full_horizon / horizon)
    forecasts = []

    for _ in range(n_predicts):
        # Previsão com futr_df preenchido
        step_forecast = nf_loaded.predict(df=combined_train, futr_df=futr_df[['unique_id', 'ds'] + calendar_cols])

        # Identificar e renomear a coluna de previsão
        forecast_column = step_forecast.columns[-1]
        step_forecast = step_forecast.rename(columns={forecast_column: 'y'})

        step_forecast = step_forecast.reset_index()
        combined_train = pd.concat([combined_train, step_forecast[['unique_id', 'ds', 'y'] + calendar_cols]],
                                   ignore_index=True)
        forecasts.append(step_forecast)

    full_forecast = pd.concat(forecasts, ignore_index=True)
    full_forecast['model_name'] = model_name

    # Salvando previsões em arquivo CSV
    forecast_output_path = os.path.join(model_output_path, f'{applied_model_name}_{model_name}_full_forecast.csv')
    full_forecast.to_csv(forecast_output_path, index=False)
    print(f"Previsões salvas em {forecast_output_path}")

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
    print(f"Métricas salvas em {log_output_path}")

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
    Y_df = Y_df.dropna(subset=['ds'])
    Y_df['y'] = Y_df['TEMPERATURA DO AR BULBO SECO (°C)']
    Y_df['unique_id'] = 'serie_1'
    Y_df = Y_df.dropna(subset=['y']).sort_values('ds').reset_index(drop=True)

    # Configuração do modelo AutoVanilla
    config = {
        "input_size": 24,
        "hidden_size": 16,
        "conv_hidden_size": 32,
        "n_head": 2,
        "learning_rate": 1e-3,
        "scaler_type": 'robust',
        "max_steps": 500
    }

    model_name = 'AutoVanilla_model'
    forecast, metrics = train_and_predict_auto_vanilla(Y_df, horizon, config, output_path, full_horizon, model_name)

    # Plot comparativo
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

    # Previsões do modelo AutoVanilla
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



