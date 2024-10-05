import pandas as pd

# Carregar os dados
data = pd.read_csv('../data/dados1975-2015.csv', sep=',', decimal=',')

# Corrigir valores inválidos na coluna 'ano'
# Exemplo: substituir valores NaN por 1975
data['ano'].fillna(1975, inplace=True)

# Converter 'ano' para inteiro
data['ano'] = data['ano'].astype(int)

# Criar coluna 'unique_id'
data['unique_id'] = 1

# Criar coluna 'ds' (sem adicionar o dia '01')
data['ds'] = pd.to_datetime(data[['ano', 'mes']].astype(str).apply(lambda x: '-'.join(x), axis=1), format='%Y-%m')

# Selecionar apenas as colunas 'unique_id', 'ds' e 'chuva'
data = data[['unique_id', 'ds', 'chuva']]
data = data.rename(columns={'chuva': 'y'})

# Salvar os dados processados com o nome "data_processed_chuva.csv"
data.to_csv('../data/data_processed_chuva.csv', sep=';', index=False)

print(data.head())