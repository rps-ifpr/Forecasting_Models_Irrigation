import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("data\data_est_local1.CSV", delimiter=';')

print("Available columns in the DataFrame:")
print(df.columns)


target_variable = "Temperatura do ar bulbo seco (°C)"

target_variable_display = "Air Temperature Dry Bulb(°C)"

if target_variable not in df.columns:
    print(f"Error: The variable '{target_variable}' was not found in the DataFrame.")
else:
    df[target_variable] = pd.to_numeric(df[target_variable], errors='coerce')

    if 'Unnamed: 19' in df.columns:
        df = df.drop(columns=['Unnamed: 19'])

    numeric_columns = df.select_dtypes(include=[np.number])

    correlations = numeric_columns.corr()[target_variable]

    print(f"Correlation with the target variable ({target_variable_display}):")
    print(correlations.sort_values(ascending=False))

    correlations_without_target = correlations.drop(index=target_variable)

    plt.figure(figsize=(10, 6))
    correlations_without_target.sort_values(ascending=False).plot(kind="bar", color="skyblue")

    plt.axhline(1.0, color='red', linestyle='--', linewidth=1, label=f'Correlation with {target_variable_display}')

    plt.title("Correlation of Variables with the Target Variable")
    plt.xlabel("Variables")
    plt.ylabel("Correlation Coefficient")
    plt.xticks(rotation=45, ha="right")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()



