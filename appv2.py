import os
import pandas as pd
import matplotlib.pyplot as plt

data_folder = "./data/v2"

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

comparison_data = []
metrics = []
distribution_data = {}
cross_validation_metrics = []

for file in csv_files:
    file_path = os.path.join(data_folder, file)

    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        continue

    df = pd.read_csv(file_path)
    df['ds'] = pd.to_datetime(df['ds'])

    model_name = df['model_name'].iloc[0]
    y_values = df['y'].values
    comparison_data.append({"Model": model_name, "ds": df['ds'], "y": y_values})

    mean_value = y_values.mean()
    std_dev = y_values.std()
    metrics.append({"Model": model_name, "Mean": mean_value, "Std Dev": std_dev})

    distribution_data[model_name] = y_values

    cv_file_path = os.path.join(data_folder, file.replace("_full_forecast.csv", "_cv_metrics.csv"))
    if os.path.exists(cv_file_path):
        cv_metrics_df = pd.read_csv(cv_file_path)
        mean_rmse = cv_metrics_df['RMSE'].mean()
        std_rmse = cv_metrics_df['RMSE'].std()
        cross_validation_metrics.append({
            "Model": model_name,
            "Mean RMSE": mean_rmse,
            "Std RMSE": std_rmse
        })
    else:
        print(f"Cross-validation results not found for {file}")

metrics_df = pd.DataFrame(metrics)
cv_metrics_df = pd.DataFrame(cross_validation_metrics)

plt.figure(figsize=(12, 6))
plt.bar(metrics_df["Model"], metrics_df["Mean"], yerr=metrics_df["Std Dev"], capsize=5)
plt.xlabel('Models')
plt.ylabel('Mean Values')
plt.title('Mean and Standard Deviation of Models')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 8))
for data in comparison_data:
    plt.plot(data["ds"], data["y"], label=data["Model"], marker='o')
plt.xlabel('Date')
plt.ylabel('Values')
plt.title('Time Series Comparison Between Models')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 8))
for model_name, values in distribution_data.items():
    plt.hist(values, bins=20, alpha=0.5, label=model_name)
plt.xlabel('Predicted Values')
plt.ylabel('Frequency')
plt.title('Distribution of Predicted Values by Model')
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 8))
plt.boxplot(distribution_data.values(), labels=distribution_data.keys(), vert=False)
plt.xlabel('Predicted Values')
plt.ylabel('Models')
plt.title('Boxplot of Predicted Values by Model')
plt.tight_layout()
plt.show()

if not cv_metrics_df.empty:
    plt.figure(figsize=(12, 6))
    plt.bar(cv_metrics_df["Model"], cv_metrics_df["Mean RMSE"], yerr=cv_metrics_df["Std RMSE"], capsize=5)
    plt.xlabel('Models')
    plt.ylabel('Mean RMSE (Cross-Validation)')
    plt.title('Mean and Standard Deviation of RMSE by Model (Cross-Validation)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

fold_results = {}
for file in csv_files:
    cv_file_path = os.path.join(data_folder, file.replace("_full_forecast.csv", "_cv_metrics.csv"))
    if os.path.exists(cv_file_path):
        cv_metrics_df = pd.read_csv(cv_file_path)
        model_name = file.split("_")[0]
        fold_results[model_name] = cv_metrics_df['RMSE']

if fold_results:
    plt.figure(figsize=(12, 8))
    plt.boxplot(fold_results.values(), labels=fold_results.keys(), vert=False)
    plt.xlabel('RMSE')
    plt.ylabel('Models')
    plt.title('Boxplot of RMSE by Model (Cross-Validation)')
    plt.tight_layout()
    plt.show()
