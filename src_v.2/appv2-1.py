import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Dados do DataFrame
data = {
    "Model": [
        "Former", "Bitcn", "Deepar", "Dilatedrnn", "Gru", "iTransformer",
        "Rnn", "Tcn", "Tft", "Fedformer", "Informer", "Lstm", "Patchtst", "VanillaTransformer"
    ],
    "RMSE": [
        1.9315098126022487, 19.627571275658628, 12.044608525508744, 12.108568551898989, 9.60784783737621,
        1.4272335185215317, 10.796256218712143, 24.051505409055416, 5.203650936292757, 2.8052750765355783,
        3.188164464371858, 9.79182580758985, 1.4635609751164236, 2.5703661382958574
    ],
    "RMSPE": [
        0.8441568495473625, 8.084254176940805, 5.170058167321451, 4.902048845405912, 3.875520133512856,
        0.6648892304913313, 4.966579630169935, 9.715218123569992, 2.3302372686152424, 1.0704471538377651,
        1.5126178530206171, 3.946486711403901, 0.6671046562814272, 1.193212393417165
    ],
    "Max Abs Error": [
        3.2869003295898445, 23.012585735321046, 18.921210479736327, 16.73994369506836, 14.370639038085937,
        2.864663696289064, 16.059724426269533, 28.95999882221222, 7.178529357910158, 5.944310760498048,
        5.819049453735353, 17.322055053710937, 3.5392070770263686, 4.724282836914064
    ],
    "Mean Abs Error": [
        1.69476531346639, 19.42304819822312, 11.747612953186035, 11.728271579742431, 9.135587120056153,
        1.244765408833822, 10.51661500930786, 23.6131536602974, 4.86469051361084, 2.1889233748118087,
        2.879516776402792, 9.010443210601808, 1.117909161249797, 2.2933887481689457
    ],
    "Median Abs Error": [
        1.4400180816650394, 20.408874988555908, 11.743130111694336, 11.926073455810547, 8.48453826904297,
        1.206803321838379, 10.044329071044922, 24.75999611020088, 5.322082328796387, 1.6633073806762706,
        2.9719097137451165, 7.816189002990722, 0.9997117996215829, 2.364911270141602
    ]
}
df = pd.DataFrame(data)


# Função para gerar gráficos com gradiente
def plot_metric_with_gradient(df, metric, cmap='viridis'):
    """
    Gera um gráfico de barras horizontais com gradiente para uma métrica específica.

    Parâmetros:
    - df: DataFrame contendo os dados.
    - metric: Métrica a ser visualizada.
    - cmap: Mapa de cores para o gradiente.
    """
    # Ordenar o DataFrame pela métrica
    df_sorted = df.sort_values(by=metric)
    values = df_sorted[metric]

    # Normalização para o gradiente de cores
    norm = plt.Normalize(vmin=values.min(), vmax=values.max())
    scalarmap = plt.cm.ScalarMappable(norm=norm, cmap=cmap)

    # Configuração do gráfico
    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.barh(df_sorted["Model"], values, color=scalarmap.to_rgba(values))
    ax.set_xlabel(metric, fontsize=12)
    ax.set_ylabel("Model", fontsize=12)
    ax.set_title(f"Comparison of Models by {metric}", fontsize=14, fontweight='bold')

    # Adicionar barra de cores
    cbar = fig.colorbar(scalarmap, ax=ax, orientation='vertical', pad=0.02)
    cbar.ax.set_ylabel(f"{metric} (Scale)", rotation=270, labelpad=20, fontsize=10)

    # Ajustes finais
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    fig.tight_layout()
    plt.show()


# Gerar gráficos para cada métrica
metrics = ["RMSE", "RMSPE", "Max Abs Error", "Mean Abs Error", "Median Abs Error"]
for metric in metrics:
    plot_metric_with_gradient(df, metric)


