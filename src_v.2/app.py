import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

# Configurar backend interativo para PyCharm
matplotlib.use('module://backend_interagg')

# Ler o arquivo gerado
data_path = '../data/generated_data_models.csv'
data = pd.read_csv(data_path)

# Converter a coluna "Date" para formato de data
data['Date'] = pd.to_datetime(data['Date'])

# Lista de modelos
models = [
    "iTransformer", "PatchTST", "FEDformer", "Autoformer", "Informer", "TFT",
    "VanillaTransformer", "BiTCN", "DilatedRNN", "DeepAR", "TCN", "GRU", "RNN", "LSTM"
]

# Gerar gráficos individuais para cada modelo
for model in models:
    plt.figure(figsize=(10, 6))

    # Plotar os dados do modelo
    plt.plot(data['Date'], data[model], label=f"{model}", color='blue', linewidth=1.5)

    # Plotar o "Ground Truth"
    plt.plot(data['Date'], data['GroundTruth'], label="Ground Truth", color='black', linestyle='--', linewidth=2)

    # Configurações do gráfico
    plt.title(f"Evolução Anual: {model} vs Ground Truth (2024)", fontsize=14)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Values", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    # Exibir o gráfico no PyCharm
    plt.show()

