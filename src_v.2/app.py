import pandas as pd
import matplotlib.pyplot as plt
import unicodedata

def normalizar_colunas(colunas):
    colunas_normalizadas = []
    for col in colunas:

        col_normalizada = ''.join(
            c for c in unicodedata.normalize('NFD', col)
            if unicodedata.category(c) != 'Mn'
        )

        col_normalizada = ' '.join(col_normalizada.upper().strip().split())
        colunas_normalizadas.append(col_normalizada)
    return colunas_normalizadas

try:

    dados = pd.read_csv('../data/data_est_local.csv', sep=';', encoding='utf-8')  # Ajuste o encoding se necessário


    dados.columns = normalizar_colunas(dados.columns)
    print("Colunas disponíveis no CSV após normalização:", dados.columns.tolist())


    colunas_necessarias = ['DATA', 'HORA', 'TEMPERATURA DO AR BULBO SECO (°C)']
    if not all(coluna in dados.columns for coluna in colunas_necessarias):
        raise ValueError(f"O arquivo CSV deve conter as colunas: {colunas_necessarias}")


    dados['ds'] = pd.to_datetime(dados['DATA'] + ' ' + dados['HORA'], dayfirst=True, errors='coerce')


    if dados['ds'].isnull().any():
        raise ValueError("Algumas datas e horas não puderam ser convertidas. Verifique o formato das colunas 'DATA' e 'HORA'.")


    dados_filtrados = dados[['ds', 'TEMPERATURA DO AR BULBO SECO (°C)']].rename(columns={
        'TEMPERATURA DO AR BULBO SECO (°C)': 'Temperatura'
    })


    dados_filtrados.dropna(subset=['Temperatura'], inplace=True)


    dados_filtrados['Temperatura'] = pd.to_numeric(dados_filtrados['Temperatura'], errors='coerce')


    dados_filtrados.dropna(subset=['Temperatura'], inplace=True)


    plt.figure(figsize=(12, 6))
    plt.plot(dados_filtrados['ds'], dados_filtrados['Temperatura'], label='Temperatura (°C)', color='blue', linewidth=1)
    plt.title('Temperatura do Ar (Bulbo Seco) ao Longo do Tempo', fontsize=14)
    plt.xlabel('Tempo', fontsize=12)
    plt.ylabel('Temperatura (°C)', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.gcf().autofmt_xdate()
    plt.tight_layout()

    plt.show()

except FileNotFoundError:
    print("Erro: O arquivo '../data/data_est_local.csv' não foi encontrado.")
except ValueError as e:
    print(f"Erro: {e}")
except Exception as e:
    print(f"Ocorreu um erro inesperado: {e}")

