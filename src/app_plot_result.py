import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

# Caminho para os arquivos de previsão e métricas
output_path = './output'

# Função para carregar um arquivo de previsão se existir
def load_forecast(file_path, col_name):
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        return df.rename(columns={'y': col_name})
    else:
        print(f"Arquivo não encontrado: {file_path}")
        return None

# Carregar arquivos de previsão, se existirem
lstm_forecast = load_forecast(f'{output_path}/checkpoints/LSTM_LSTM_model/LSTM_LSTM_model_full_forecast.csv', 'LSTM_Prediction')
autotft_forecast = load_forecast(f'{output_path}/checkpoints/AutoTFT_AutoTFT_model/AutoTFT_AutoTFT_model_full_forecast.csv', 'AutoTFT_Prediction')
autornn_forecast = load_forecast(f'{output_path}/checkpoints/AutoRNN_AutoRNN_model/AutoRNN_AutoRNN_model_full_forecast.csv', 'AutoRNN_Prediction')
autoinformer_forecast = load_forecast(f'{output_path}/checkpoints/AutoInformer_AutoInformer_model/AutoInformer_AutoInformer_model_full_forecast.csv', 'AutoInformer_Prediction')
vanilla_forecast = load_forecast(f'{output_path}/checkpoints/VanillaTransformer_VanillaTransformer_model/VanillaTransformer_VanillaTransformer_model_full_forecast.csv', 'VanillaTransformer_Prediction')
autobitcn_forecast = load_forecast(f'{output_path}/checkpoints/AutoBiTCN_AutoBiTCN_model/AutoBiTCN_AutoBiTCN_model_full_forecast.csv', 'AutoBiTCN_Prediction')
autodeepar_forecast = load_forecast(f'{output_path}/checkpoints/AutoDeepAR_AutoDeepAR_model/AutoDeepAR_AutoDeepAR_model_full_forecast.csv', 'AutoDeepAR_Prediction')
autodilatedrnn_forecast = load_forecast(f'{output_path}/checkpoints/AutoDilatedRNN_AutoDilatedRNN_model/AutoDilatedRNN_AutoDilatedRNN_model_full_forecast.csv', 'AutoDilatedRNN_Prediction')
autogru_forecast = load_forecast(f'{output_path}/checkpoints/AutoGRU_AutoGRU_model/AutoGRU_AutoGRU_model_full_forecast.csv', 'AutoGRU_Prediction')
autotcn_forecast = load_forecast(f'{output_path}/checkpoints/AutoTCN_AutoTCN_model/AutoTCN_AutoTCN_model_full_forecast.csv', 'AutoTCN_Prediction')

# Filtrar previsões carregadas com sucesso para combinar no DataFrame
forecasts = [df for df in [lstm_forecast, autotft_forecast, autornn_forecast, autoinformer_forecast, vanilla_forecast, autobitcn_forecast, autodeepar_forecast, autodilatedrnn_forecast, autogru_forecast, autotcn_forecast] if df is not None]

# Combinar previsões em um único DataFrame, usando a coluna 'ds' como referência e ignorando colunas duplicadas
comparison_df = forecasts[0][['ds']].copy()
for forecast in forecasts:
    comparison_df = pd.merge(comparison_df, forecast.drop(columns=['unique_id', 'model_name'], errors='ignore'), on='ds', how='inner')

# Exibir a tabela comparativa das previsões para análise
print("\nTabela comparativa das previsões dos modelos:")
print(comparison_df.head(20))  # Exibir as primeiras 20 linhas para visualização

# Gráfico de Previsões Comparativas
plt.figure(figsize=(14, 8))
for col in comparison_df.columns[1:]:
    plt.plot(comparison_df['ds'], comparison_df[col], label=col)
plt.xlabel('Data')
plt.ylabel('Previsão')
plt.title('Comparação das Previsões dos Modelos')
plt.legend()
plt.grid(True)
plt.gca().xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Carregar arquivos de métricas para cada modelo, se existirem
def load_metrics(file_path, model_name):
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        df['Modelo'] = model_name
        return df
    else:
        print(f"Arquivo de métricas não encontrado: {file_path}")
        return None

# Carregar métricas dos modelos
metrics_dfs = [
    load_metrics(f'{output_path}/lightning_logs/Autoformer_Autoformer_model/Autoformer_Autoformer_model_metrics.csv', 'Autoformer'),
    load_metrics(f'{output_path}/lightning_logs/AutoBiTCN_AutoBiTCN_model/AutoBiTCN_AutoBiTCN_model_metrics.csv', 'AutoBiTCN'),
    load_metrics(f'{output_path}/lightning_logs/AutoDeepAR_AutoDeepAR_model/AutoDeepAR_AutoDeepAR_model_metrics.csv', 'AutoDeepAR'),
    load_metrics(f'{output_path}/lightning_logs/AutoDilatedRNN_AutoDilatedRNN_model/AutoDilatedRNN_AutoDilatedRNN_model_metrics.csv', 'AutoDilatedRNN'),
    load_metrics(f'{output_path}/lightning_logs/AutoGRU_AutoGRU_model/AutoGRU_AutoGRU_model_metrics.csv', 'AutoGRU'),
    load_metrics(f'{output_path}/lightning_logs/AutoInformer_AutoInformer_model/AutoInformer_AutoInformer_model_metrics.csv', 'AutoInformer'),
    load_metrics(f'{output_path}/lightning_logs/AutoRNN_AutoRNN_model/AutoRNN_AutoRNN_model_metrics.csv', 'AutoRNN'),
    load_metrics(f'{output_path}/lightning_logs/AutoTCN_AutoTCN_model/AutoTCN_AutoTCN_model_metrics.csv', 'AutoTCN'),
    load_metrics(f'{output_path}/lightning_logs/AutoTFT_AutoTFT_model/AutoTFT_AutoTFT_model_metrics.csv', 'AutoTFT'),
    load_metrics(f'{output_path}/lightning_logs/FEDformer_AutoFEDformer_model/FEDformer_AutoFEDformer_model_metrics.csv', 'FEDformer'),
    load_metrics(f'{output_path}/lightning_logs/Informer_AutoInformer_model/Informer_AutoInformer_model_metrics.csv', 'Informer'),
    load_metrics(f'{output_path}/lightning_logs/LSTM_LSTM_model/LSTM_LSTM_model_metrics.csv', 'LSTM'),
    load_metrics(f'{output_path}/lightning_logs/PatchTST_AutoPatchTST_model/PatchTST_AutoPatchTST_model_metrics.csv', 'PatchTST'),
    load_metrics(f'{output_path}/lightning_logs/VanillaTransformer_VanillaTransformer_model/VanillaTransformer_VanillaTransformer_model_metrics.csv', 'VanillaTransformer')
]

# Filtrar métricas carregadas com sucesso para combinar no DataFrame
metrics_df = pd.concat([df for df in metrics_dfs if df is not None], ignore_index=True)

# Configurar o DataFrame para exibir métricas lado a lado
metrics_df = metrics_df.set_index(['Modelo', 'model_name']).T  # Transpor para facilitar a comparação

# Gráfico de Métricas dos Modelos com rótulos
fig, ax = plt.subplots(figsize=(14, 8))
metrics_df.plot(kind='bar', ax=ax)
plt.title('Comparação das Métricas - Modelos')
plt.xlabel('Métricas')
plt.ylabel('Valores')

# Adicionar rótulos de valores acima de cada barra
for p in ax.patches:
    ax.annotate(f'{p.get_height():.2f}',
                (p.get_x() + p.get_width() / 2, p.get_height()),
                ha='center', va='bottom', fontsize=8)

plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(title='Modelos', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
