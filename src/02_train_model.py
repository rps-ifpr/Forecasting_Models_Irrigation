import time
import pandas as pd
import numpy as np
from neuralforecast.auto import AutoTFT
from neuralforecast.losses.pytorch import MAE
from ray import tune
from neuralforecast.core import NeuralForecast

def train_model(data_path, horizon, config, output_path):
    """Treina e salva o modelo."""

    # Load the data
    Y_df = pd.read_csv(data_path, sep=';')

    # Set the threshold until the max
    max = np.where(Y_df.y == np.max(Y_df.y))
    Y_df = Y_df[0:max[-1][-1]]

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


if __name__ == '__main__':
    data_path = '../data/data.csv'
    horizon = 1
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
    train_model(data_path, horizon, config, output_path)