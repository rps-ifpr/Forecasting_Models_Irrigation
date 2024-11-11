import os
import pandas as pd
import matplotlib.pyplot as plt

output_path = './output'

def load_cross_val_metrics(file_path, model_name):
    """Carrega métricas de validação cruzada para um modelo específico."""
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        if 'Model' not in df.columns:
            df['Model'] = model_name  # Adiciona a coluna Model caso não exista
        return df
    else:
        print(f"Metrics file not found: {file_path}")
        return pd.DataFrame()  # Retorna DataFrame vazio se o arquivo não existir

# Carregar métricas de validação cruzada
metrics_dfs = [
    load_cross_val_metrics(f'{output_path}/lightning_logs/Autoformer_Autoformer_model/Autoformer_Autoformer_model_metrics.csv', 'Autoformer'),
    load_cross_val_metrics(f'{output_path}/lightning_logs/AutoBiTCN_AutoBiTCN_model/AutoBiTCN_AutoBiTCN_model_metrics.csv', 'AutoBiTCN'),
    load_cross_val_metrics(f'{output_path}/lightning_logs/AutoDeepAR_AutoDeepAR_model/AutoDeepAR_AutoDeepAR_model_metrics.csv', 'AutoDeepAR'),
    load_cross_val_metrics(f'{output_path}/lightning_logs/AutoDilatedRNN_AutoDilatedRNN_model/AutoDilatedRNN_AutoDilatedRNN_model_metrics.csv', 'AutoDilatedRNN'),
    load_cross_val_metrics(f'{output_path}/lightning_logs/AutoGRU_AutoGRU_model/AutoGRU_AutoGRU_model_metrics.csv', 'AutoGRU'),
    load_cross_val_metrics(f'{output_path}/lightning_logs/iTransformer_AutoiTransformer_model/iTransformer_AutoiTransformer_model_metrics.csv', 'iTransformer'),
    load_cross_val_metrics(f'{output_path}/lightning_logs/AutoRNN_AutoRNN_model/AutoRNN_AutoRNN_model_metrics.csv', 'AutoRNN'),
    load_cross_val_metrics(f'{output_path}/lightning_logs/AutoTCN_AutoTCN_model/AutoTCN_AutoTCN_model_metrics.csv', 'AutoTCN'),
    load_cross_val_metrics(f'{output_path}/lightning_logs/AutoTFT_AutoTFT_model/AutoTFT_AutoTFT_model_metrics.csv', 'AutoTFT'),
    load_cross_val_metrics(f'{output_path}/lightning_logs/FEDformer_AutoFEDformer_model/FEDformer_AutoFEDformer_model_metrics.csv', 'FEDformer'),
    load_cross_val_metrics(f'{output_path}/lightning_logs/Informer_AutoInformer_model/Informer_AutoInformer_model_metrics.csv', 'Informer'),
    load_cross_val_metrics(f'{output_path}/lightning_logs/LSTM_LSTM_model/LSTM_LSTM_model_metrics.csv', 'LSTM'),
    load_cross_val_metrics(f'{output_path}/lightning_logs/PatchTST_AutoPatchTST_model/PatchTST_AutoPatchTST_model_metrics.csv', 'PatchTST'),
    load_cross_val_metrics(f'{output_path}/lightning_logs/VanillaTransformer_VanillaTransformer_model/VanillaTransformer_VanillaTransformer_model_metrics.csv', 'VanillaTransformer')
]

# Verificar se os DataFrames foram carregados corretamente
valid_metrics_dfs = [df for df in metrics_dfs if not df.empty]
if not valid_metrics_dfs:
    print("Nenhum arquivo de métricas foi carregado com sucesso. Verifique os caminhos e o conteúdo dos arquivos.")
    exit()

# Concatenar métricas
metrics_df = pd.concat(valid_metrics_dfs, ignore_index=True)

# Garantir que a coluna 'Model' está presente
if 'Model' not in metrics_df.columns:
    print("A coluna 'Model' está ausente nos dados carregados. Verifique os arquivos de métricas.")
    exit()

# Salvar as métricas consolidadas
metrics_df.to_csv('consolidated_model_metrics.csv', index=False)

# Configurar índice e transpor para visualização
metrics_df = metrics_df.set_index('Model')

# Plotar as métricas
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

fig, ax = plt.subplots(figsize=(12, 8))
metrics_df.plot(kind='barh', ax=ax, color=colors[:len(metrics_df.columns)], width=0.85)
plt.title('Metric Comparison - Models')
plt.xlabel('Values (Mean)')
plt.ylabel('Metrics')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.legend(title='Metrics', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


