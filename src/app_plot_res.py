import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

# Caminho para os arquivos de previsão e métricas
output_path = './output'

# Leitura dos arquivos de previsão
lstm_forecast = pd.read_csv(f'{output_path}/checkpoints/LSTM_LSTM_model/LSTM_LSTM_model_full_forecast.csv')
autotft_forecast = pd.read_csv(f'{output_path}/checkpoints/AutoTFT_AutoTFT_model/AutoTFT_AutoTFT_model_forecast.csv')
autornn_forecast = pd.read_csv(f'{output_path}/checkpoints/AutoRNN_AutoRNN_model/AutoRNN_AutoRNN_model_full_forecast.csv')

# Leitura dos arquivos de métricas
lstm_metrics = pd.read_csv(f'{output_path}/lightning_logs/LSTM_LSTM_model/LSTM_LSTM_model_metrics.csv')
autotft_metrics = pd.read_csv(f'{output_path}/lightning_logs/AutoTFT_AutoTFT_model/AutoTFT_AutoTFT_model_metrics.csv')
autornn_metrics = pd.read_csv(f'{output_path}/lightning_logs/AutoRNN_AutoRNN_model/AutoRNN_AutoRNN_model_metrics.csv')

# Renomear as colunas 'y' para identificar cada modelo
lstm_forecast = lstm_forecast.rename(columns={'y': 'LSTM_Prediction'})
autotft_forecast = autotft_forecast.rename(columns={'y': 'AutoTFT_Prediction'})
autornn_forecast = autornn_forecast.rename(columns={'y': 'AutoRNN_Prediction'})

# Combinar as previsões dos três modelos na mesma tabela para comparação
comparison_df = pd.merge(lstm_forecast[['ds', 'LSTM_Prediction']],
                         autotft_forecast[['ds', 'AutoTFT_Prediction']], on='ds', how='inner')
comparison_df = pd.merge(comparison_df, autornn_forecast[['ds', 'AutoRNN_Prediction']], on='ds', how='inner')

# Exibir a tabela comparativa das previsões para análise
print("\nTabela comparativa das previsões dos modelos LSTM, AutoTFT e AutoRNN:")
print(comparison_df.head(20))  # Exibir as primeiras 20 linhas para visualização

# Gráfico de Previsões Comparativas
plt.figure(figsize=(12, 6))
plt.plot(comparison_df['ds'], comparison_df['LSTM_Prediction'], label='LSTM Prediction', color='blue')
plt.plot(comparison_df['ds'], comparison_df['AutoTFT_Prediction'], label='AutoTFT Prediction', color='orange')
plt.plot(comparison_df['ds'], comparison_df['AutoRNN_Prediction'], label='AutoRNN Prediction', color='green')
plt.xlabel('Data')
plt.ylabel('Previsão')
plt.title('Comparação das Previsões dos Modelos LSTM, AutoTFT e AutoRNN')
plt.legend()
plt.grid(True)
plt.gca().xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Gráfico de Métricas dos Modelos
metrics_df = pd.concat([
    lstm_metrics.assign(Modelo='LSTM'),
    autotft_metrics.assign(Modelo='AutoTFT'),
    autornn_metrics.assign(Modelo='AutoRNN')
])

# Configurar o DataFrame para exibir métricas lado a lado
metrics_df = metrics_df.set_index(['Modelo', 'model_name']).T  # Transpor para facilitar a comparação

# Plot das métricas com rótulos em cada barra
fig, ax = plt.subplots(figsize=(12, 6))
metrics_df.plot(kind='bar', ax=ax, color=['blue', 'orange', 'green'])
plt.title('Comparação das Métricas - Modelos LSTM, AutoTFT e AutoRNN')
plt.xlabel('Métricas')
plt.ylabel('Valores')

# Adicionar rótulos de valores acima de cada barra
for p in ax.patches:
    ax.annotate(f'{p.get_height():.2f}',
                (p.get_x() + p.get_width() / 2, p.get_height()),
                ha='center', va='bottom', fontsize=10)

plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(title='Modelos')
plt.tight_layout()
plt.show()

