import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from tqdm import trange

from certified_removal import certified_removal
from logistic_pru import logistic_pru

df = pd.read_csv("data/prepared_loan.csv")
feature = df.drop(columns=["Keputusan Akhir"]).to_numpy()
target = df["Keputusan Akhir"].to_numpy()

# Experiment parameters
# Modify as needed
test_size = (0.2, 0.3, 0.4)  # 0.2, 0.3, 0.4
lambda_ = (0.001, 0.005, 0.01)

test_size_count = len(test_size)  # 3
lambda_count = len(lambda_)  # 3
removal_count = 16  # 12, 14, 16
repetition = 100  # 100
dimension = 23  # original_weights.shape[1] = 23
# End of experiment parameters

classifier = LogisticRegressionCV(Cs=101)

weights = {
    "naive": np.zeros((test_size_count, lambda_count, removal_count, repetition, dimension)),
    "cr": np.zeros((test_size_count, lambda_count, removal_count, repetition, dimension)),
    "pru": np.zeros((test_size_count, lambda_count, removal_count, repetition, dimension)),
}
accuracy = {
    "naive": np.zeros((test_size_count, lambda_count, removal_count, repetition)),
    "cr": np.zeros((test_size_count, lambda_count, removal_count, repetition)),
    "pru": np.zeros((test_size_count, lambda_count, removal_count, repetition)),
}
l2_distance = {
    "base": np.zeros((test_size_count, lambda_count, removal_count, repetition)),
    "cr": np.zeros((test_size_count, lambda_count, removal_count, repetition)),
    "pru": np.zeros((test_size_count, lambda_count, removal_count, repetition)),
}

for p in trange(test_size_count):
    current_test_size = test_size[p]
    original_model = joblib.load(f"models/benchmark-classifier-{current_test_size}.joblib")
    original_weights = original_model.coef_
    original_intercept = original_model.intercept_
    for q in trange(lambda_count, leave=False):
        current_lambda = lambda_[q]
        for r in trange(removal_count, leave=False):
            for s in trange(repetition, leave=False):
                train_feature, test_feature, train_target, test_target = train_test_split(feature, target, test_size=current_test_size)

                classifier.fit(train_feature[r + 1 :], train_target[r + 1 :])
                retrained_weights = classifier.coef_
                weights["naive"][p][q][r][s] = retrained_weights.squeeze()
                accuracy["naive"][p][q][r][s] = classifier.score(test_feature, test_target)

                weights_after_cr = certified_removal(r, train_feature, train_target, original_weights, current_lambda)
                weights["cr"][p][q][r][s] = weights_after_cr.squeeze()

                prediction_after_cr = test_feature @ np.transpose(weights_after_cr) + original_intercept
                accuracy_after_cr = (np.squeeze((prediction_after_cr > 0.5)) == (test_target > 0.5)).astype(float).mean()
                accuracy["cr"][p][q][r][s] = accuracy_after_cr

                weights_after_pru = logistic_pru(r, train_feature, train_target, original_weights, current_lambda)
                weights["pru"][p][q][r][s] = weights_after_pru.squeeze()

                prediction_after_pru = test_feature @ np.transpose(weights_after_pru) + original_intercept
                accuracy_after_pru = (np.squeeze((prediction_after_pru > 0.5)) == (test_target > 0.5)).astype(float).mean()
                accuracy["pru"][p][q][r][s] = accuracy_after_pru

    l2_distance["base"][p] = np.linalg.norm(weights["naive"][p] - original_weights, axis=3)
    l2_distance["cr"][p] = np.linalg.norm(weights["naive"][p] - weights["cr"][p], axis=3) / l2_distance["base"][p]
    l2_distance["pru"][p] = np.linalg.norm(weights["naive"][p] - weights["pru"][p], axis=3) / l2_distance["base"][p]

# joblib.dump(weights, "out/rev/weights-latest.joblib")
# joblib.dump(accuracy, "out/rev/accuracy-latest.joblib")
# joblib.dump(l2_distance, "out/rev/l2_distance-latest.joblib")
