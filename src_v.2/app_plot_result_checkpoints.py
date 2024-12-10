import os
import pandas as pd
import matplotlib.pyplot as plt
import logging

# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

output_path = './output'

def load_cross_val_metrics(model_name):
    """Carrega métricas de validação cruzada para um modelo específico a partir de uma estrutura de diretório padronizada."""
    file_path = f'{output_path}/lightning_logs/{model_name}_{model_name}_model/{model_name}_{model_name}_model_metrics.csv'
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        df['Model'] = model_name  # Garante que a coluna Model está presente
        return df
    else:
        logging.warning(f"Metrics file not found: {file_path}")
        return pd.DataFrame()

model_names = ['Autoformer', 'AutoBiTCN', 'AutoDeepAR', 'AutoDilatedRNN', 'AutoGRU',
               'iTransformer', 'AutoRNN', 'AutoTCN', 'AutoTFT', 'FEDformer',
               'Informer', 'LSTM', 'PatchTST', 'VanillaTransformer']

# Carregar métricas de validação cruzada
metrics_dfs = [load_cross_val_metrics(model) for model in model_names]

# Concatenar métricas e verificar se algum DataFrame foi carregado
metrics_df = pd.concat(metrics_dfs, ignore_index=True)
if metrics_df.empty:
    logging.error("Nenhum arquivo de métricas foi carregado com sucesso. Verifique os caminhos e o conteúdo dos arquivos.")
    exit()

# Salvar as métricas consolidadas
metrics_df.to_csv('consolidated_model_metrics.csv', index=False)

# Configuração de plotagem
fig, ax = plt.subplots(figsize=(12, 8))
metrics_df.set_index('Model').plot(kind='barh', ax=ax, width=0.85)
plt.title('Metric Comparison - Models')
plt.xlabel('Values (Mean)')
plt.ylabel('Models')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.legend(title='Metrics', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


