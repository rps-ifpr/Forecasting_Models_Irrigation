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


def train_and_predict(data_path, horizon, config, output_path, full_horizon):
    """Treina o modelo, salva e faz previsões."""

    # Carrega e prepara os dados
    Y_df = pd.read_csv(data_path, sep=';')
    Y_df['ds'] = pd.to_datetime(Y_df['Data'] + ' ' + Y_df['Hora'])
    Y_df['y'] = Y_df['TEMPERATURA DO AR BULBO SECO (°C)']
    Y_df = Y_df.dropna(subset=['y']).sort_values('ds').reset_index(drop=True)

    # Seleção até o valor máximo de `y`
    max_idx = np.where(Y_df.y == np.max(Y_df.y))[0][-1]
    Y_df = Y_df.iloc[:max_idx + 1]

    # Plotagem dos dados originais
    plt.figure(figsize=(10, 5))
    plt.plot(Y_df['ds'], Y_df['y'], 'k')
    plt.xlabel('Data')
    plt.ylabel('Temperatura do Ar (°C)')
    plt.title('Série Temporal da Temperatura do Ar')
    plt.grid(True)
    plt.show()

    # Treinamento do modelo
    start_time = time.time()
    models = [AutoTFT(h=horizon, loss=MAE(), config=config, num_samples=10)]
    nf = NeuralForecast(models=models, freq='H')

    nf.fit(df=Y_df[['ds', 'y']])
    nf.save(path=output_path, model_index=None, overwrite=True, save_dataset=True)
    end_time = time.time()
    print('Tempo para treinar o modelo:', end_time - start_time)

    # Carregando o modelo treinado
    nf_loaded = NeuralForecast.load(path=output_path)

    # Faz previsões
    Y_hat_df = nf_loaded.predict(df=Y_df[['ds', 'y']])
    y_pred = Y_hat_df.AutoTFT.values
    y_true = Y_df['y'][-horizon:].values

    # Métricas de erro em notação científica
    print(
        f'{rmse(y_true, y_pred):.2E} & {rmspe(y_true, y_pred):.2E} & '
        f'{maxabs(y_true, y_pred):.2E} & {meanabs(y_true, y_pred):.2E} & {medianabs(y_true, y_pred):.2E}'
    )

    # Previsão de múltiplos passos com redefinição de índice
    n_predicts = math.ceil(full_horizon / models[0].h)
    combined_train = Y_df[['ds', 'y']].copy()
    forecasts = []
    for _ in range(n_predicts):
        step_forecast = nf_loaded.predict(df=combined_train)
        forecasts.append(step_forecast)
        step_forecast = step_forecast.rename(columns={'AutoTFT': 'y'})
        combined_train = pd.concat([combined_train, step_forecast[['ds', 'y']]], ignore_index=True)

    # Combina todas as previsões e plota
    final_forecast = pd.concat(forecasts, ignore_index=True)
    plt.figure(figsize=(10, 5))
    plt.plot(Y_df['ds'], Y_df['y'], label='Dados Reais')
    plt.plot(final_forecast['ds'], final_forecast['AutoTFT'], label='Previsões', linestyle='--')
    plt.xlabel('Data')
    plt.ylabel('Temperatura do Ar (°C)')
    plt.title('Previsão da Temperatura do Ar')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    data_path = '../data/data_est_local.csv'  # Atualize este caminho
    horizon = 1  # Consistente com o primeiro código
    full_horizon = 20  # Previsão para múltiplos passos
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
    output_path = 'checkpoints/test_run/'

    train_and_predict(data_path, horizon, config, output_path, full_horizon)
