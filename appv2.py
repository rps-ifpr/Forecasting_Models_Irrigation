import os
import pandas as pd
import matplotlib.pyplot as plt

# Pasta com os arquivos CSV
data_folder = "./data/v2"

# Lista de arquivos a serem analisados
csv_files = [
    "iTransformer_AutoiTransformer_model_full_forecast.csv",
    "PatchTST_AutoPatchTST_model_full_forecast.csv",
    "FEDformer_AutoFEDformer_model_full_forecast.csv",
    "Autoformer_Autoformer_model_full_forecast.csv",
    "Informer_AutoInformer_model_full_forecast.csv",
    "AutoTFT_AutoTFT_model_full_forecast.csv",
    "VanillaTransformer_VanillaTransformer_model_full_forecast.csv",
    "AutoBiTCN_AutoBiTCN_model_full_forecast.csv",
    "AutoDilatedRNN_AutoDilatedRNN_model_full_forecast.csv",
    "AutoDeepAR_AutoDeepAR_model_full_forecast.csv",
    "AutoTCN_AutoTCN_model_full_forecast.csv",
    "AutoGRU_AutoGRU_model_full_forecast.csv",
    "AutoRNN_AutoRNN_model_full_forecast.csv",
    "LSTM_LSTM_model_full_forecast.csv",
]

comparison_data = []  # Dados para o gráfico de séries temporais
metrics = []  # Dados para as métricas
distribution_data = {}  # Dados para o histograma e boxplot

# Processar cada arquivo individualmente
for file in csv_files:
    file_path = os.path.join(data_folder, file)

    # Verificar se o arquivo existe
    if not os.path.exists(file_path):
        print(f"Arquivo não encontrado: {file_path}")
        continue

    # Carregar os dados do CSV
    df = pd.read_csv(file_path)
    df['ds'] = pd.to_datetime(df['ds'])  # Certificar-se de que a coluna 'ds' está no formato datetime

    # Adicionar ao gráfico combinado e coletar dados
    model_name = df['model_name'].iloc[0]  # Nome do modelo
    y_values = df['y'].values
    comparison_data.append({"Model": model_name, "ds": df['ds'], "y": y_values})

    # Calcular métricas básicas
    mean_value = y_values.mean()
    std_dev = y_values.std()
    metrics.append({"Model": model_name, "Mean": mean_value, "Std Dev": std_dev})

    # Adicionar dados para o histograma e boxplot
    distribution_data[model_name] = y_values

# Criar gráficos

# 1. Gráfico de média e desvio padrão
metrics_df = pd.DataFrame(metrics)
plt.figure(figsize=(12, 6))
plt.bar(metrics_df["Model"], metrics_df["Mean"], yerr=metrics_df["Std Dev"], capsize=5)
plt.xlabel('Modelos')
plt.ylabel('Média dos Valores')
plt.title('Média e Desvio Padrão dos Modelos')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# 2. Gráfico de séries temporais
plt.figure(figsize=(12, 8))
for data in comparison_data:
    plt.plot(data["ds"], data["y"], label=data["Model"], marker='o')
plt.xlabel('Data')
plt.ylabel('Valores')
plt.title('Comparação de Séries Temporais Entre Modelos')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# 3. Histograma de distribuição
plt.figure(figsize=(12, 8))
for model_name, values in distribution_data.items():
    plt.hist(values, bins=20, alpha=0.5, label=model_name)
plt.xlabel('Valores Previstos')
plt.ylabel('Frequência')
plt.title('Distribuição dos Valores Previstos por Modelo')
plt.legend()
plt.tight_layout()
plt.show()

# 4. Boxplot
plt.figure(figsize=(12, 8))
plt.boxplot(distribution_data.values(), labels=distribution_data.keys(), vert=False)
plt.xlabel('Valores Previstos')
plt.ylabel('Modelos')
plt.title('Boxplot dos Valores Previstos por Modelo')
plt.tight_layout()
plt.show()






