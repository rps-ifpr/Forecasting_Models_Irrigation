import os
import pandas as pd
import matplotlib.pyplot as plt

# Pasta com os arquivos CSV
data_folder = "./data/"

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

# Lista para armazenar dados para comparação
comparison_data = []

# Processar cada arquivo individualmente
for file in csv_files:
    file_path = os.path.join(data_folder, file)

    # Verificar se o arquivo existe
    if not os.path.exists(file_path):
        print(f"Arquivo não encontrado: {file_path}")
        continue

    # Carregar os dados do CSV
    df = pd.read_csv(file_path)

    # Certificar-se de que a coluna 'ds' está no formato datetime
    df['ds'] = pd.to_datetime(df['ds'])

    # Adicionar os dados do modelo
    model_name = df['model_name'].iloc[0]  # Extrair o nome do modelo
    comparison_data.append({
        "Model": model_name,
        "ds": df['ds'],
        "y": df['y']
    })

    # Plotar gráfico individual
    plt.figure(figsize=(10, 6))
    plt.plot(df['ds'], df['y'], label=f'Valores Reais ({model_name})', marker='o')
    plt.xlabel('Data')
    plt.ylabel('Valores')
    plt.title(f'Valores Reais ao Longo do Tempo - {model_name}')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

plt.figure(figsize=(12, 8))
for data in comparison_data:
    plt.plot(data["ds"], data["y"], label=f'Valores Reais - {data["Model"]}')
plt.xlabel('Data')
plt.ylabel('Valores')
plt.title('Comparação de Valores Reais Entre Modelos')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()




