import os
import pandas as pd
import matplotlib.pyplot as plt

output_path = './output'

def load_metrics(file_path, model_name):
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        df['Model'] = model_name
        return df
    else:
        print(f"Metrics file not found: {file_path}")
        return None

metrics_dfs = [
    load_metrics(f'{output_path}/lightning_logs/Autoformer_Autoformer_model/Autoformer_Autoformer_model_metrics.csv', 'Autoformer'),
    load_metrics(f'{output_path}/lightning_logs/AutoBiTCN_AutoBiTCN_model/AutoBiTCN_AutoBiTCN_model_metrics.csv', 'AutoBiTCN'),
    load_metrics(f'{output_path}/lightning_logs/AutoDeepAR_AutoDeepAR_model/AutoDeepAR_AutoDeepAR_model_metrics.csv', 'AutoDeepAR'),
    load_metrics(f'{output_path}/lightning_logs/AutoDilatedRNN_AutoDilatedRNN_model/AutoDilatedRNN_AutoDilatedRNN_model_metrics.csv', 'AutoDilatedRNN'),
    load_metrics(f'{output_path}/lightning_logs/AutoGRU_AutoGRU_model/AutoGRU_AutoGRU_model_metrics.csv', 'AutoGRU'),
    load_metrics(f'{output_path}/lightning_logs/iTransformer_AutoiTransformer_model/iTransformer_AutoiTransformer_model_metrics.csv', 'AutoInformer'),
    load_metrics(f'{output_path}/lightning_logs/AutoRNN_AutoRNN_model/AutoRNN_AutoRNN_model_metrics.csv', 'AutoRNN'),
    load_metrics(f'{output_path}/lightning_logs/AutoTCN_AutoTCN_model/AutoTCN_AutoTCN_model_metrics.csv', 'AutoTCN'),
    load_metrics(f'{output_path}/lightning_logs/AutoTFT_AutoTFT_model/AutoTFT_AutoTFT_model_metrics.csv', 'AutoTFT'),
    load_metrics(f'{output_path}/lightning_logs/FEDformer_AutoFEDformer_model/FEDformer_AutoFEDformer_model_metrics.csv', 'FEDformer'),
    load_metrics(f'{output_path}/lightning_logs/Informer_AutoInformer_model/Informer_AutoInformer_model_metrics.csv', 'Informer'),
    load_metrics(f'{output_path}/lightning_logs/LSTM_LSTM_model/LSTM_LSTM_model_metrics.csv', 'LSTM'),
    load_metrics(f'{output_path}/lightning_logs/PatchTST_AutoPatchTST_model/PatchTST_AutoPatchTST_model_metrics.csv', 'PatchTST'),
    load_metrics(f'{output_path}/lightning_logs/VanillaTransformer_VanillaTransformer_model/VanillaTransformer_VanillaTransformer_model_metrics.csv', 'VanillaTransformer')
]

metrics_df = pd.concat([df for df in metrics_dfs if df is not None], ignore_index=True)
metrics_df = metrics_df.set_index(['Model', 'model_name']).T

rnn_based_models = ['AutoRNN', 'LSTM', 'AutoGRU', 'AutoTCN', 'AutoDeepAR', 'AutoDilatedRNN', 'AutoBiTCN']
transformer_based_models = ['AutoTFT', 'VanillaTransformer', 'AutoInformer', 'Autoformer', 'FEDformer', 'PatchTST', 'Informer']

colors = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    '#aec7e8', '#ffbb78', '#98df8a', '#ff9896'
]

fig, ax = plt.subplots(figsize=(12, 8))
metrics_df.plot(kind='barh', ax=ax, color=colors[:len(metrics_df.columns)], width=0.85)
plt.title('Metric Comparison - Models')
plt.xlabel('Values')
plt.ylabel('Metrics')

for p in ax.patches:
    ax.annotate(f'{p.get_width():.2f}',
                (p.get_width(), p.get_y() + p.get_height() / 2),
                ha='left', va='center', fontsize=8, color='black', xytext=(5, 0), textcoords='offset points')

plt.subplots_adjust(left=0.2, right=0.85, top=0.9, bottom=0.1)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.legend(title='Models', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

for metric in metrics_df.index:
    sorted_metric_df = metrics_df.loc[metric].sort_values()
    fig, ax = plt.subplots(figsize=(10, 6))
    sorted_metric_df.plot(kind='barh', color=colors[:len(sorted_metric_df)], width=0.85, ax=ax)
    plt.title(f'Metric Comparison - {metric} (Ascending Order)')
    plt.xlabel('Value')
    plt.ylabel('Model')

    for p in ax.patches:
        ax.annotate(f'{p.get_width():.2f}',
                    (p.get_width(), p.get_y() + p.get_height() / 2),
                    ha='left', va='center', fontsize=8, color='black', xytext=(5, 0), textcoords='offset points')

    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

for metric in metrics_df.index:
    rnn_metric_df = metrics_df.loc[metric, [(model, metric) for model in rnn_based_models if (model, metric) in metrics_df.columns]].sort_values()
    if not rnn_metric_df.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        rnn_metric_df.plot(kind='barh', color=colors[:len(rnn_metric_df)], width=0.85, ax=ax)
        plt.title(f'RNN-Based Metric Comparison - {metric} (Ascending Order)')
        plt.xlabel('Value')
        plt.ylabel('Model')

        for p in ax.patches:
            ax.annotate(f'{p.get_width():.2f}',
                        (p.get_width(), p.get_y() + p.get_height() / 2),
                        ha='left', va='center', fontsize=8, color='black', xytext=(5, 0), textcoords='offset points')

        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

    transformer_metric_df = metrics_df.loc[metric, [(model, metric) for model in transformer_based_models if (model, metric) in metrics_df.columns]].sort_values()
    if not transformer_metric_df.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        transformer_metric_df.plot(kind='barh', color=colors[:len(transformer_metric_df)], width=0.85, ax=ax)
        plt.title(f'Transformer-Based Metric Comparison - {metric} (Ascending Order)')
        plt.xlabel('Value')
        plt.ylabel('Model')

        for p in ax.patches:
            ax.annotate(f'{p.get_width():.2f}',
                        (p.get_width(), p.get_y() + p.get_height() / 2),
                        ha='left', va='center', fontsize=8, color='black', xytext=(5, 0), textcoords='offset points')

        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()





