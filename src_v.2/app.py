import pandas as pd
import matplotlib.pyplot as plt
import unicodedata

def normalize_columns(columns):
    normalized_columns = []
    for col in columns:
        normalized_col = ''.join(
            c for c in unicodedata.normalize('NFD', col)
            if unicodedata.category(c) != 'Mn'
        )
        normalized_col = ' '.join(normalized_col.upper().strip().split())
        normalized_columns.append(normalized_col)
    return normalized_columns

try:
    data = pd.read_csv('../data/data_est_local.csv', sep=';', encoding='utf-8')

    data.columns = normalize_columns(data.columns)
    print("Available columns in the CSV after normalization:", data.columns.tolist())

    required_columns = ['DATA', 'HORA', 'TEMPERATURA DO AR BULBO SECO (°C)']
    if not all(column in data.columns for column in required_columns):
        raise ValueError(f"The CSV file must contain the columns: {required_columns}")

    data['ds'] = pd.to_datetime(data['DATA'] + ' ' + data['HORA'], dayfirst=True, errors='coerce')

    if data['ds'].isnull().any():
        raise ValueError("Some dates and times could not be converted. Check the format of the 'DATA' and 'HORA' columns.")

    filtered_data = data[['ds', 'TEMPERATURA DO AR BULBO SECO (°C)']].rename(columns={
        'TEMPERATURA DO AR BULBO SECO (°C)': 'Temperature'
    })

    filtered_data.dropna(subset=['Temperature'], inplace=True)

    filtered_data['Temperature'] = pd.to_numeric(filtered_data['Temperature'], errors='coerce')

    filtered_data.dropna(subset=['Temperature'], inplace=True)

    plt.figure(figsize=(12, 6))
    plt.plot(filtered_data['ds'], filtered_data['Temperature'], label='Temperature (°C)', color='blue', linewidth=1)
    plt.title('Air Temperature (Dry Bulb) Over Time', fontsize=14)
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Temperature (°C)', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.gcf().autofmt_xdate()
    plt.tight_layout()

    plt.show()

except FileNotFoundError:
    print("Error: The file '../data/data_est_local.csv' was not found.")
except ValueError as e:
    print(f"Error: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

