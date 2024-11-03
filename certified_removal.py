import joblib
import numpy as np
import pandas as pd
import scipy

LAMBDA = 1e-3


def invert_hessian(weight_array, feature_array, target_array, lambda_=1e-3, batch_size=50000):
    """
    Calculate the inverse hessian of the model's loss function.
    :param weight_array: The model's weight.
    :param feature_array: The model's features.
    :param target_array: The model's targets.
    :param lambda_: The regularization parameter.
    :param batch_size: The size of the removal batch.
    :return: The inverse hessian of the model's loss function.
    """
    z_matrix = scipy.special.expit(feature_array @ np.transpose(weight_array) @ np.expand_dims(target_array, 0))
    d_matrix = z_matrix * (1 - z_matrix)
    hessian_matrix = None
    batch_count = int(np.ceil(feature_array.shape[0] / batch_size))
    for i in range(batch_count):
        lower = i * batch_size
        upper = min((i + 1) * batch_size, feature_array.shape[0])
        targeted_feature = feature_array[lower:upper]
        if hessian_matrix is None:
            hessian_matrix = np.transpose(targeted_feature) @ d_matrix[lower:upper] @ targeted_feature
        else:
            hessian_matrix += np.transpose(targeted_feature) @ d_matrix[lower:upper] @ targeted_feature
    return np.linalg.inv(hessian_matrix + lambda_ * feature_array.shape[0] * np.eye(feature_array.shape[1]))


def gradient(weight_array, feature_array, target_array, lambda_=1e-3):
    """
    Calculate the gradient of the model's loss function.
    :param weight_array: The model's weight.
    :param feature_array: The model's features.
    :param target_array: The model's targets.
    :param lambda_: The regularization parameter.
    :return: The gradient of the model's loss function.
    """
    z_matrix = scipy.special.expit(feature_array @ np.transpose(weight_array) @ np.expand_dims(target_array, 0))
    # fmt: off
    return (np.transpose(feature_array) @ ((z_matrix - 1) * target_array)
            + lambda_ * feature_array.shape[0] * np.transpose(weight_array))
    # fmt: on


def certified_removal(removal_count, data_feature, data_target, original_weights, lambda_=LAMBDA):
    """
    Perform the certified removal procedure on a given model.
    :param removal_count: The number of data points to be removed.
    :param data_feature: The data set features.
    :param data_target: The data set targets.
    :param original_weights: The data set weights.
    :param lambda_: The regularization parameter.
    :return: The updated model weights after certified removal.
    """
    updated_weights = original_weights.copy()

    for i in range(removal_count):
        kept_feature = data_feature[i + 1 :]
        kept_target = data_target[i + 1 :]
        inverse_hessian = invert_hessian(updated_weights[:], kept_feature, kept_target, lambda_)
        current_gradient = gradient(
            updated_weights,
            np.expand_dims(data_feature[i], 0),
            np.expand_dims(data_target[i], 0),
            lambda_,
        )
        delta = inverse_hessian @ current_gradient
        updated_weights += np.transpose(delta)

    return updated_weights


if __name__ == "__main__":
    k = 5
    k_word = "five"

    data = pd.read_csv("data/prepared_loan.csv")
    leftover = pd.read_csv(f"data/leave_{k_word}_out.csv")
    naive_model = joblib.load("models/naive-classifier.joblib")
    weights = naive_model.coef_
    intercept = naive_model.intercept_

    removed_df = pd.concat([data, leftover]).drop_duplicates(keep=False)
    ordered_df = pd.concat([removed_df, leftover], ignore_index=True)
    feature = ordered_df.drop(columns=["Keputusan Akhir"]).to_numpy()
    target = ordered_df["Keputusan Akhir"].to_numpy()

    weights_after_removal = certified_removal(k, feature, target, weights)
    print(f"Weights after removal: {weights_after_removal}")

    lko_model = joblib.load(f"models/leave-{k_word}-out-classifier.joblib")
    lko_weights = lko_model.coef_
    lko_intercept = lko_model.intercept_

    l2_distance = lko_weights - weights_after_removal
    print(f"L2-distance: {l2_distance}")
    print(f"L2-distance mean: {abs(l2_distance).mean()}")

    prediction_before = feature @ np.transpose(lko_weights) + lko_intercept
    prediction_after = feature @ np.transpose(weights_after_removal) + intercept
    accuracy_before = (np.squeeze((prediction_before > 0.5), 1) == (target > 0.5)).astype(float).mean()
    accuracy_after = (np.squeeze((prediction_after > 0.5), 1) == (target > 0.5)).astype(float).mean()
    print(f"Accuracy before removal: {accuracy_before}")
    print(f"Accuracy after removal: {accuracy_after}")
