import os
import pandas as pd
import matplotlib.pyplot as plt

# Diretório de saída dos arquivos
output_path = './output'

def load_metrics(file_path, model_name):
    """Carrega métricas de um arquivo CSV e adiciona o nome do modelo."""
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        df['Model'] = model_name
        return df
    else:
        print(f"Metrics file not found: {file_path}")
        return None

def plot_metrics_overall(metrics_df, colors):
    """Gera gráfico de barras horizontal comparando todas as métricas entre os modelos."""
    fig, ax = plt.subplots(figsize=(12, 8))
    metrics_df.plot(kind='barh', ax=ax, color=colors[:len(metrics_df.columns)], width=0.85)
    plt.title('Metric Comparison - Models')
    plt.xlabel('Values')
    plt.ylabel('Metrics')

    # Anotações nos gráficos
    for p in ax.patches:
        ax.annotate(f'{p.get_width():.2f}',
                    (p.get_width(), p.get_y() + p.get_height() / 2),
                    ha='left', va='center', fontsize=8, color='black', xytext=(5, 0), textcoords='offset points')

    plt.subplots_adjust(left=0.2, right=0.85, top=0.9, bottom=0.1)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.legend(title='Models', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

def plot_metrics_by_model_type(metrics_df, models, colors, metric, title_prefix):
    """Gera gráficos de métricas específicas por tipo de modelo (RNN ou Transformer)."""
    metric_df = metrics_df.loc[metric, [(model, metric) for model in models if (model, metric) in metrics_df.columns]].sort_values()
    if not metric_df.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        metric_df.plot(kind='barh', color=colors[:len(metric_df)], width=0.85, ax=ax)
        plt.title(f'{title_prefix} Metric Comparison - {metric} (Ascending Order)')
        plt.xlabel('Value')
        plt.ylabel('Model')

        # Anotações nos gráficos
        for p in ax.patches:
            ax.annotate(f'{p.get_width():.2f}',
                        (p.get_width(), p.get_y() + p.get_height() / 2),
                        ha='left', va='center', fontsize=8, color='black', xytext=(5, 0), textcoords='offset points')

        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

# Modelos RNN e Transformer
rnn_based_models = ['RNN', 'LSTM', 'GRU', 'TCN', 'DeepAR', 'DilatedRNN', 'BiTCN']
transformer_based_models = ['TFT', 'VanillaTransformer', 'Informer', 'Autoformer', 'FEDformer', 'PatchTST', 'Informer']

# Cores para os gráficos
colors = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    '#aec7e8', '#ffbb78', '#98df8a', '#ff9896'
]

# Carregar métricas
metrics_dfs = [
    load_metrics(f'{output_path}/lightning_logs/Autoformer_Autoformer_model/Autoformer_Autoformer_model_metrics.csv', 'Autoformer'),
    load_metrics(f'{output_path}/lightning_logs/AutoBiTCN_AutoBiTCN_model/AutoBiTCN_AutoBiTCN_model_metrics.csv', 'BiTCN'),
    load_metrics(f'{output_path}/lightning_logs/AutoDeepAR_AutoDeepAR_model/AutoDeepAR_AutoDeepAR_model_metrics.csv', 'DeepAR'),
    load_metrics(f'{output_path}/lightning_logs/AutoDilatedRNN_AutoDilatedRNN_model/AutoDilatedRNN_AutoDilatedRNN_model_metrics.csv', 'DilatedRNN'),
    load_metrics(f'{output_path}/lightning_logs/AutoGRU_AutoGRU_model/AutoGRU_AutoGRU_model_metrics.csv', 'GRU'),
    load_metrics(f'{output_path}/lightning_logs/iTransformer_AutoiTransformer_model/iTransformer_AutoiTransformer_model_metrics.csv', 'iTransformer'),
    load_metrics(f'{output_path}/lightning_logs/AutoRNN_AutoRNN_model/AutoRNN_AutoRNN_model_metrics.csv', 'RNN'),
    load_metrics(f'{output_path}/lightning_logs/AutoTCN_AutoTCN_model/AutoTCN_AutoTCN_model_metrics.csv', 'TCN'),
    load_metrics(f'{output_path}/lightning_logs/AutoTFT_AutoTFT_model/AutoTFT_AutoTFT_model_metrics.csv', 'TFT'),
    load_metrics(f'{output_path}/lightning_logs/FEDformer_AutoFEDformer_model/FEDformer_AutoFEDformer_model_metrics.csv', 'FEDformer'),
    load_metrics(f'{output_path}/lightning_logs/Informer_AutoInformer_model/Informer_AutoInformer_model_metrics.csv', 'Informer'),
    load_metrics(f'{output_path}/lightning_logs/LSTM_LSTM_model/LSTM_LSTM_model_metrics.csv', 'LSTM'),
    load_metrics(f'{output_path}/lightning_logs/PatchTST_AutoPatchTST_model/PatchTST_AutoPatchTST_model_metrics.csv', 'PatchTST'),
    load_metrics(f'{output_path}/lightning_logs/VanillaTransformer_VanillaTransformer_model/VanillaTransformer_VanillaTransformer_model_metrics.csv', 'VanillaTransformer')
]

# Concatenar todas as métricas carregadas
metrics_df = pd.concat([df for df in metrics_dfs if df is not None], ignore_index=True)
metrics_df = metrics_df.set_index(['Model', 'model_name']).T

# Gerar gráficos de todas as métricas
plot_metrics_overall(metrics_df, colors)

# Gerar gráficos separados para RNN e Transformers por métrica
for metric in metrics_df.index:
    plot_metrics_by_model_type(metrics_df, rnn_based_models, colors, metric, 'RNN-Based')
    plot_metrics_by_model_type(metrics_df, transformer_based_models, colors, metric, 'Transformer-Based')




