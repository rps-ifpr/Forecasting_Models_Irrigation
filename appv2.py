import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
        1.69476531346639, 19.42304819822312, 11.747564938447292, 11.46691120577921, 9.095759513246165,
        1.2927063214060467, 10.204554562220054, 23.581495721685613, 4.987139587833008, 2.7654014567759517,
        3.0551685712188253, 8.977739616727306, 1.325423283780862, 2.3304339955193525
    ]
}


np.random.seed(42)
num_models = len(data["Model"])
ground_truth = np.random.uniform(low=0.5, high=3.0, size=num_models)


df = pd.DataFrame(data)
df["Ground Truth"] = ground_truth


plt.figure(figsize=(14, 7))
plt.plot(df["Model"], df["Ground Truth"], 'o-', label="Ground Truth", color="black")


bar_width = 0.2
x = np.arange(len(df["Model"]))

plt.bar(x - 2 * bar_width, df["RMSE"], bar_width, label="RMSE")
plt.bar(x - bar_width, df["RMSPE"], bar_width, label="RMSPE")
plt.bar(x, df["Max Abs Error"], bar_width, label="Max Abs Error")
plt.bar(x + bar_width, df["Mean Abs Error"], bar_width, label="Mean Abs Error")


plt.xticks(x, df["Model"], rotation=45)
plt.xlabel("Models")
plt.ylabel("Values")
plt.title("Comparison of Ground Truth and Error Metrics Across Models")
plt.legend()
plt.tight_layout()
plt.show()

