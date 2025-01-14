import pandas as pd
import numpy as np
from datetime import datetime, timedelta

start_date = datetime(2024, 1, 1)
end_date = start_date + timedelta(days=365)
dates = pd.date_range(start=start_date, end=end_date, freq='D')

# Ground truth data
ground_truth_data = np.linspace(0, 10, len(dates)) + 10 * np.sin(np.linspace(0, 10 * np.pi, len(dates)))

# Simulando previsões de modelos com diferentes padrões
models_data = {}
for i in range(1, 15):
    if i % 2 == 0:  # Modelos pares com tendência positiva
        model_data = ground_truth_data + np.random.normal(scale=2, size=len(dates))
    else:  # Modelos ímpares com tendência negativa
        model_data = ground_truth_data - np.random.normal(scale=2, size=len(dates))
    models_data[f'Model_{i}'] = model_data

# Criando o DataFrame
df = pd.DataFrame({'ds': dates, 'Ground_Truth': ground_truth_data})
for name, data in models_data.items():
    df[name] = data

# Salvar em CSV
df.to_csv('./data/v2/all_models_ground_truth.csv', index=False)



