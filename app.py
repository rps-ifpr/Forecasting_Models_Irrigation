import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Caminho absoluto para o arquivo CSV
csv_path = "./src/output/checkpoints/AutoBiTCN_AutoBiTCN_model/AutoBiTCN_AutoBiTCN_model_full_forecast.csv"

# Verifique se o arquivo existe
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"Arquivo não encontrado: {csv_path}")

# Carregar dados do CSV
df = pd.read_csv(csv_path)

# Converter a coluna 'ds' para datetime
df['ds'] = pd.to_datetime(df['ds'])

# Calcular métricas de desempenho
rmse = np.sqrt(mean_squared_error(df['y'], df['BiTCN']))
mae = mean_absolute_error(df['y'], df['BiTCN'])
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")

# Plotar Previsão vs Observado
plt.figure(figsize=(10, 6))
plt.plot(df['ds'], df['y'], label='Valores Reais (y)', marker='o')
plt.plot(df['ds'], df['BiTCN'], label='Previsão (BiTCN)', linestyle='--')
plt.fill_between(df['ds'], df['BiTCN-lo-90'], df['BiTCN-hi-90'], color='gray', alpha=0.3, label='Intervalo de Confiança 90%')
plt.fill_between(df['ds'], df['BiTCN-lo-80'], df['BiTCN-hi-80'], color='blue', alpha=0.2, label='Intervalo de Confiança 80%')
plt.xlabel('Data')
plt.ylabel('Valores')
plt.title('Previsão vs Valores Reais com Intervalos de Confiança')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# Estatísticas descritivas
print("\nEstatísticas Descritivas das Previsões:")
print(df[['BiTCN', 'y']].describe())

# Analisar dispersão (resíduos)
df['residuals'] = df['y'] - df['BiTCN']
plt.figure(figsize=(10, 6))
plt.scatter(df['ds'], df['residuals'], label='Resíduos', color='red')
plt.axhline(0, color='black', linestyle='--')
plt.xlabel('Data')
plt.ylabel('Resíduos')
plt.title('Análise de Resíduos')
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()


