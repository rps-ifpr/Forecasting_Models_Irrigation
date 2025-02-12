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
from statsmodels.tools.eval_measures import vare  # variance of error
from statsmodels.tools.eval_measures import stde  # standard deviation of error
from statsmodels.tools.eval_measures import iqr  # interquartile range of error
import math


def train_and_predict(data_path, horizon, config, output_path, full_horizon):
    """Treina o modelo, salva e faz previsões."""


    Y_df = pd.read_csv(data_path, sep=';')


    max = np.where(Y_df.y == np.max(Y_df.y))
    Y_df = Y_df[0:max[-1][-1]]


    plt.figure(figsize=(6, 3))
    plt.plot(Y_df.y, 'k')
    plt.axis([0, len(list(Y_df.y)), 0, 100])
    plt.xlabel('Time (h)')
    plt.ylabel('Level (%)')
    plt.grid(linestyle='-', which='both')
    plt.show()

    Y_df['ds'] = pd.to_datetime(Y_df['ds'], format='%d/%m/%y %H:%M')


    start_time = time.time()
    models = [AutoTFT(h=horizon,
                      loss=MAE(),
                      config=config,
                      num_samples=10)]

    nf = NeuralForecast(
        models=models,
        freq='H')

    nf.fit(df=Y_df)
    nf.save(path=output_path,
            model_index=None,
            overwrite=True,
            save_dataset=True)
    end_time = time.time()
    print('Time to train model:', end_time - start_time)


    model = AutoTFT(h=1, loss=MAE(), config=config, num_samples=10)
    nf_loaded = NeuralForecast(models=[model], freq='H')
    nf_loaded = NeuralForecast.load(path=output_path)


    Y_hat_df = nf_loaded.predict(df=Y_df)
    y_pred = Y_hat_df.AutoTFT
    y_true = Y_df[-horizon:].y

    end_time = time.time()
    print('Time to create models:', end_time - start_time)


    print(
        f'{(rmse(y_true, y_pred)):.2E} & {(rmspe(y_true, y_pred)):.2E} & {(maxabs(y_true, y_pred)):.2E} & {(meanabs(y_true, y_pred)):.2E} & {(medianabs(y_true, y_pred)):.2E} '
    )


    n_predicts = math.ceil(full_horizon / model.h)
    combined_train = Y_df
    forecasts = []
    for _ in range(n_predicts):
        step_forecast = nf_loaded.predict(df=combined_train)
        forecasts.append(step_forecast)
        step_forecast = step_forecast.rename(columns={'AutoTFT': 'y'})
        step_forecast = step_forecast.reset_index()
        combined_train = pd.concat([combined_train, step_forecast])
        combined_train = combined_train.reset_index(drop=True)
    pd.concat(forecasts)


if __name__ == '__main__':
    data_path = '../data/data.csv'
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