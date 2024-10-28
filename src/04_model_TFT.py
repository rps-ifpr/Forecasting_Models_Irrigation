import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from neuralforecast.auto import AutoTFT
from neuralforecast.losses.pytorch import MAE
from ray import tune
from neuralforecast.core import NeuralForecast
from statsmodels.tools.eval_measures import rmse  # root mean squared error
from statsmodels.tools.eval_measures import rmspe  # root mean squared percentage error
from statsmodels.tools.eval_measures import maxabs  # maximum absolute error
from statsmodels.tools.eval_measures import meanabs  # mean absolute error
from statsmodels.tools.eval_measures import medianabs  # median absolute error
import math
import os

def train_and_predict(Y_df, horizon, config, output_path, full_horizon, model_name):
    """Treina o modelo, salva e faz previsões, retorna previsões e métricas."""

    # Nome do modelo aplicado
    applied_model_name = 'AutoTFT'

    # Diretório específico para o modelo
    model_output_path = os.path.join(output_path, f"{applied_model_name}_{model_name}")
    if not os.path.exists(model_output_path):
        os.makedirs(model_output_path)

    # Treinamento do modelo
    start_time = time.time()
    models = [AutoTFT(h=horizon,
                      loss=MAE(),
                      config=config,
                      num_samples=10)]

    nf = NeuralForecast(
        models=models,
        freq='H')

    nf.fit(df=Y_df[['unique_id', 'ds', 'y']])
    nf.save(path=model_output_path,
            model_index=None,
            overwrite=True,
            save_dataset=True)
    end_time = time.time()
    print(f'Modelo {model_name} treinado em:', end_time - start_time, 'segundos')

    # Carregando o modelo treinado
    nf_loaded = NeuralForecast.load(path=model_output_path)

    # Faz previsões de múltiplos passos
    n_predicts = math.ceil(full_horizon / horizon)
    combined_train = Y_df[['unique_id', 'ds', 'y']].copy()
    forecasts = []

    for _ in range(n_predicts):
        step_forecast = nf_loaded.predict(df=combined_train)
        step_forecast = step_forecast.rename(columns={applied_model_name: 'y'})

        # Resetar o índice para trazer 'unique_id' e 'ds' como colunas
        step_forecast = step_forecast.reset_index()

        # Garantir que 'unique_id' esteja presente e seja consistente
        if 'unique_id' not in step_forecast.columns:
            step_forecast['unique_id'] = 'serie_1'
        else:
            step_forecast['unique_id'] = step_forecast['unique_id'].astype(str)

        # Concatenar previsões com o conjunto de treinamento
        combined_train = pd.concat([combined_train, step_forecast[['unique_id', 'ds', 'y']]], ignore_index=True)

        # Garantir que 'unique_id' em 'combined_train' seja do tipo string
        combined_train['unique_id'] = combined_train['unique_id'].astype(str)

        forecasts.append(step_forecast)

    # Concatenar todas as previsões
    full_forecast = pd.concat(forecasts, ignore_index=True)
    full_forecast['model_name'] = model_name  # Adicionar coluna com o nome do modelo

    # Salvar as previsões em um arquivo CSV
    forecast_output_path = os.path.join(model_output_path, f'{applied_model_name}_{model_name}_full_forecast.csv')
    full_forecast.to_csv(forecast_output_path, index=False)

    # Calcular métricas
    y_pred = full_forecast['y'].values
    y_true = Y_df['y'].iloc[-len(y_pred):].values  # Valores reais correspondentes
    rmse_value = rmse(y_true, y_pred)
    rmspe_value = rmspe(y_true, y_pred)
    maxabs_value = maxabs(y_true, y_pred)
    meanabs_value = meanabs(y_true, y_pred)
    medianabs_value = medianabs(y_true, y_pred)

    # Retornar previsões e métricas
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

    # Carregar os dados fora da função para que Y_df esteja disponível no escopo global
    Y_df = pd.read_csv(data_path, sep=';')

    # Combinar as colunas 'Data' e 'Hora' em uma única coluna datetime
    Y_df['ds'] = pd.to_datetime(Y_df['Data'] + ' ' + Y_df['Hora'], format='%Y-%m-%d %H:%M:%S')

    # Selecionar a variável alvo e renomeá-la para 'y'
    Y_df['y'] = Y_df['TEMPERATURA DO AR BULBO SECO (°C)']

    # Adicionar a coluna 'unique_id'
    Y_df['unique_id'] = 'serie_1'

    # Remover linhas com valores ausentes na variável alvo
    Y_df = Y_df.dropna(subset=['y'])

    # Ordenar o DataFrame por datetime
    Y_df = Y_df.sort_values('ds').reset_index(drop=True)

    # Nome do modelo aplicado
    applied_model_name = 'AutoTFT'

    # Lista de modelos e configurações
    models_to_test = [
        {
            'model_name': 'modelo_inicial',
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
        # Você pode adicionar mais modelos aqui
    ]

    all_forecasts = []
    all_metrics = []

    for model_info in models_to_test:
        model_name = model_info['model_name']
        config = model_info['config']
        print(f'Treinando e avaliando o modelo {applied_model_name} com configuração {model_name}')
        forecast, metrics = train_and_predict(Y_df, horizon, config, output_path, full_horizon, model_name)
        all_forecasts.append(forecast)
        all_metrics.append(metrics)

    # Concatenar previsões de todos os modelos
    combined_forecasts = pd.concat(all_forecasts, ignore_index=True)

    # Plotar previsões de todos os modelos em um único gráfico
    plt.figure(figsize=(12, 6))
    plt.plot(Y_df['ds'], Y_df['y'], label='Dados Originais', color='black')
    for model_name in combined_forecasts['model_name'].unique():
        model_forecast = combined_forecasts[combined_forecasts['model_name'] == model_name]
        plt.plot(model_forecast['ds'], model_forecast['y'], label=f'Previsão {model_name}')
    plt.xlabel('Tempo')
    plt.ylabel('Temperatura (°C)')
    plt.title('Comparação das Previsões dos Modelos')
    plt.legend()
    plt.grid(linestyle='-', which='both')
    plt.show()

    # Criar um DataFrame com as métricas de todos os modelos
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.set_index('model_name', inplace=True)

    # Plotar métricas de todos os modelos em um único gráfico
    metrics_df.plot(kind='bar', figsize=(12, 6))
    plt.title('Comparação das Métricas dos Modelos')
    plt.xlabel('Modelos')
    plt.ylabel('Valores')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()


