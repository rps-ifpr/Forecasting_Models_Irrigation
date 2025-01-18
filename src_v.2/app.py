import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

dates = pd.date_range(start='2023-11-01', end='2023-12-30', freq='D')
num_points = len(dates)

np.random.seed(42)
ground_truth = 25 + 5 * np.sin(2 * np.pi * np.arange(num_points) / 30) + np.random.normal(0, 2, num_points)

np.random.seed(42)
performance = {
    'iTransformer': ground_truth + np.random.normal(0, 1.5, num_points),
    'PatchTST': ground_truth + 1 + np.random.normal(0, 2.0, num_points),
    'FEDformer': ground_truth - 1 + np.random.normal(0, 1.0, num_points),
    'Autoformer': ground_truth + 2 + np.random.normal(0, 2.5, num_points),
    'Informer': ground_truth - 2 + np.random.normal(0, 2.2, num_points),
    'TFT': ground_truth + 3 + np.random.normal(0, 1.8, num_points),
    'VanillaTransformer': ground_truth - 3 + np.random.normal(0, 2.0, num_points),
    'BiTCN': ground_truth + np.linspace(0, 3, num_points) + np.random.normal(0, 2.2, num_points),
    'DilatedRNN': ground_truth - np.linspace(0, 3, num_points) + np.random.normal(0, 2.3, num_points),
    'DeepAR': ground_truth + np.linspace(0, 2, num_points) + np.random.normal(0, 1.5, num_points),
    'TCN': ground_truth - np.linspace(0, 2, num_points) + np.random.normal(0, 1.8, num_points),
    'GRU': ground_truth + 0.5 + np.random.normal(0, 1.2, num_points),
    'RNN': ground_truth - 0.5 + np.random.normal(0, 1.0, num_points),
    'LSTM': ground_truth + np.linspace(-1, 1, num_points) + np.random.normal(0, 1.5, num_points),
}

data = pd.DataFrame(index=dates, data=performance)

data['Ground Truth'] = ground_truth

plt.figure(figsize=(14, 8))

colors = cm.gray(np.linspace(0.3, 1, len(performance)))

for color, (model_name, model_data) in zip(colors, performance.items()):
    plt.plot(data.index, model_data, color=color, label=model_name, linewidth=1.5, alpha=0.8)

plt.plot(data.index, data['Ground Truth'], color='red', label='Ground Truth', linewidth=2.5, linestyle='--')

plt.title("Tendências dos Modelos com Ground Truth", fontsize=16)
plt.xlabel("Data", fontsize=12)
plt.ylabel("Indice", fontsize=12)
plt.legend(loc="upper left", fontsize=9, ncol=2)
plt.grid(True, linestyle='--', linewidth=0.5)
plt.tight_layout()

plt.show()
