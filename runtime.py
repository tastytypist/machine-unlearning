from timeit import timeit

import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegressionCV
from tqdm import tqdm

from certified_removal import certified_removal
from logistic_pru import logistic_pru, calculate_hessian

# Experiment parameters
# Modify as needed
removal_experiment = 20  # 20
repetition = 100  # 100
trial = 5
# End of experiment parameters

df = pd.read_csv("data/prepared_loan.csv")

original_model = joblib.load("models/naive-classifier.joblib")
original_weights = original_model.coef_

classifier = LogisticRegressionCV(Cs=np.logspace(-5, 5, num=101))

runtime = {
    "naive": np.zeros((removal_experiment, repetition)),
    "cr": np.zeros((removal_experiment, repetition)),
    "pru": np.zeros((removal_experiment, repetition)),
}


def naive_runtime(k):
    return classifier.fit(feature[k:], target[k:])


def cr_runtime(k):
    return certified_removal(k, feature, target, original_weights)


def pru_runtime(k):
    return logistic_pru(k, feature, target, original_weights, input_hessian=hessian_matrix)


for i in tqdm(range(1, removal_experiment + 1)):
    for j in tqdm(range(repetition), leave=False):
        df = df.sample(frac=1)
        feature = df.drop(columns=["Keputusan Akhir"]).to_numpy()
        target = df["Keputusan Akhir"].to_numpy()
        hessian_matrix = calculate_hessian(feature, original_weights)

        runtime["naive"][i - 1][j] = (
            timeit(lambda: naive_runtime(i), setup="from __main__ import naive_runtime", number=trial) / trial
        )
        runtime["cr"][i - 1][j] = (
            timeit(lambda: cr_runtime(i), setup="from __main__ import cr_runtime", number=trial) / trial
        )
        runtime["pru"][i - 1][j] = (
            timeit(lambda: pru_runtime(i), setup="from __main__ import pru_runtime", number=trial) / trial
        )

# joblib.dump(runtime, "out/runtime.joblib")
# joblib.dump(runtime, "out/runtime_pru.joblib")
