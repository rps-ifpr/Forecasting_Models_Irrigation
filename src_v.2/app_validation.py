import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dates = pd.date_range(start='2023-01-01', end='2023-12-31')
num_days = len(dates)

np.random.seed(42)
performance = {
    'Autoformer': np.random.normal(loc=1.93, scale=0.2, size=num_days),
    'BiTCN': np.random.normal(loc=19.63, scale=1.5, size=num_days),
    'DeepAR': np.random.normal(loc=12.04, scale=0.5, size=num_days),
    'DilatedRNN': np.random.normal(loc=12.11, scale=0.7, size=num_days),
    'GRU': np.random.normal(loc=9.61, scale=0.8, size=num_days),
    '.Informer': np.random.normal(loc=1.43, scale=0.1, size=num_days),
    'RNN': np.random.normal(loc=10.80, scale=1.0, size=num_days),
    'TCN': np.random.normal(loc=24.05, scale=2.0, size=num_days),
    'TFT': np.random.normal(loc=5.20, scale=0.6, size=num_days),
    'FEDformer': np.random.normal(loc=2.81, scale=0.3, size=num_days),
    'Informer': np.random.normal(loc=3.19, scale=0.4, size=num_days),
    'LSTM': np.random.normal(loc=9.79, scale=0.9, size=num_days),
    'PatchTST': np.random.normal(loc=1.46, scale=0.05, size=num_days),
    'VanillaTransformer': np.random.normal(loc=2.57, scale=0.3, size=num_days)
}

data = pd.DataFrame(index=dates, data=performance)

fig, axs = plt.subplots((len(data.columns) + 1) // 2, 2, figsize=(15, 20))
fig.subplots_adjust(hspace=0.5, wspace=0.3)
axs = axs.ravel()

for i, (model_name, model_data) in enumerate(data.items()):
    axs[i].plot(data.index, model_data, label='Model Performance')
    axs[i].set_title(f"Model Performance {model_name}")
    axs[i].set_xlabel("Date")
    axs[i].set_ylabel("Simulated Error")
    axs[i].grid(True)

    end_train = pd.to_datetime('2023-06-30')
    end_validation = pd.to_datetime('2023-09-30')
    axs[i].axvline(end_train, color='red', linestyle='--', label='End of Training')
    axs[i].axvline(end_validation, color='green', linestyle='--', label='End of Validation')
    axs[i].legend(loc='upper right')

plt.tight_layout()
plt.show()






