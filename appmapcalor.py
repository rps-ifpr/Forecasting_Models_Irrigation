import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Dados
models = ['Autoformer', 'BiTCN', 'DeepAR', 'DilatedRNN', 'GRU', 'Informer', 'RNN', 'TCN', 'TFT', 'FEDformer', 'Informer', 'LSTM', 'PatchTST', 'VanillaT_former']
metrics = ['RMSE', 'RMSPE', 'MaxAbsE', 'MeanAbsE', 'MedianAbsE']
data = np.array([
    [1.93, 0.84, 3.29, 1.69, 1.44],
    [19.63, 8.08, 23.01, 19.42, 20.41],
    [12.04, 5.17, 18.92, 11.75, 11.74],
    [12.11, 4.90, 16.74, 11.73, 11.93],
    [9.61, 3.88, 14.37, 9.14, 8.48],
    [1.43, 0.66, 2.86, 1.24, 1.21],
    [10.80, 4.97, 16.06, 10.52, 10.04],
    [24.05, 9.72, 28.96, 23.61, 24.76],
    [5.20, 2.33, 7.18, 4.86, 5.32],
    [2.81, 1.07, 5.94, 2.19, 1.66],
    [3.19, 1.51, 5.82, 2.88, 2.97],
    [9.79, 3.95, 17.32, 9.01, 7.82],
    [1.46, 0.67, 3.54, 1.12, 1.00],
    [2.57, 1.19, 4.72, 2.29, 2.36]
])

# Cria um DataFrame para melhor manipulação dos dados
df = pd.DataFrame(data, index=models, columns=metrics)

# Cria o gráfico de calor
plt.figure(figsize=(11, 8))
sns.heatmap(df, annot=True, fmt=".2f", cmap='viridis', linewidths=.5, linecolor='gray')
plt.title('Heatmap of Model Performance Metrics')
plt.xlabel('Metrics')
plt.ylabel('Models')

plt.show()

