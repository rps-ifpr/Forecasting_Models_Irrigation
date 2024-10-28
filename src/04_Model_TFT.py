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


def train_and_predict(data_path, horizon, config, output_path, full_horizon):
    """Treina o modelo, salva e faz previsões."""

    # Carregar os dados
    Y_df = pd.read_csv(data_path, sep=';')

    # Combinar as colunas 'Data' e 'Hora' em uma única coluna datetime
    Y_df['ds'] = pd.to_datetime(Y_df['Data'] + ' ' + Y_df['Hora'], format='%Y-%m-%d %H:%M:%S')

    # Selecionar a variável alvo (por exemplo, 'TEMPERATURA DO AR BULBO SECO (°C)') e renomeá-la para 'y'
    Y_df['y'] = Y_df['TEMPERATURA DO AR BULBO SECO (°C)']

    # Adicionar a coluna 'unique_id'
    Y_df['unique_id'] = 'serie_1'

    # Remover linhas com valores ausentes na variável alvo
    Y_df = Y_df.dropna(subset=['y'])

    # Ordenar o DataFrame por datetime
    Y_df = Y_df.sort_values('ds').reset_index(drop=True)

    # Plota os dados originais
    plt.figure(figsize=(10, 5))
    plt.plot(Y_df['ds'], Y_df['y'], 'k')
    plt.xlabel('Time')
    plt.ylabel('Temperature (°C)')
    plt.title('Temperatura ao longo do tempo')
    plt.grid(linestyle='-', which='both')
    plt.show()

    # Treinamento do modelo
    start_time = time.time()
    models = [AutoTFT(h=horizon,
                      loss=MAE(),
                      config=config,
                      num_samples=10)]

    nf = NeuralForecast(
        models=models,
        freq='H')

    # Ajuste: Passar as colunas 'unique_id', 'ds' e 'y' para o método fit
    nf.fit(df=Y_df[['unique_id', 'ds', 'y']])
    nf.save(path=output_path,
            model_index=None,
            overwrite=True,
            save_dataset=True)
    end_time = time.time()
    print('Tempo para treinar o modelo:', end_time - start_time)
    print(f'Modelo salvo em: {output_path}')

    # Carregando o modelo treinado
    model = AutoTFT(h=1, loss=MAE(), config=config, num_samples=10)
    nf_loaded = NeuralForecast(models=[model], freq='H')
    nf_loaded = NeuralForecast.load(path=output_path)

    # Faz previsões
    Y_hat_df = nf_loaded.predict(df=Y_df[['unique_id', 'ds', 'y']])
    y_pred = Y_hat_df.AutoTFT
    y_true = Y_df[-horizon:].y

    end_time = time.time()
    print('Tempo para criar previsões:', end_time - start_time)

    # Métricas de erro
    print(
        f'RMSE: {(rmse(y_true, y_pred)):.2E}, RMSPE: {(rmspe(y_true, y_pred)):.2E}, '
        f'Max Abs Error: {(maxabs(y_true, y_pred)):.2E}, Mean Abs Error: {(meanabs(y_true, y_pred)):.2E}, '
        f'Median Abs Error: {(medianabs(y_true, y_pred)):.2E}'
    )

    # Previsões de múltiplos passos
    n_predicts = math.ceil(full_horizon / model.h)
    combined_train = Y_df[['unique_id', 'ds', 'y']].copy()
    forecasts = []

    for _ in range(n_predicts):
        step_forecast = nf_loaded.predict(df=combined_train)
        step_forecast = step_forecast.rename(columns={'AutoTFT': 'y'})

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
    print(full_forecast)


if __name__ == '__main__':
    data_path = '../data/data_est_local.csv'
    horizon = 1
    full_horizon = 20
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
