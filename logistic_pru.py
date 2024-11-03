import joblib
import numpy as np
import pandas as pd
import scipy

LAMBDA = 1e-3


def hessian(feature_array, s_array, lambda_=1e-3):
    """
    Compute the Hessian matrix of the model's loss function.
    :param feature_array: The model's features.
    :param s_array: The matrix of logits in diagonal form.
    :param lambda_: The regularization parameter.
    :return: The Hessian matrix of the model's loss function.
    """
    symmetric_matrix = np.transpose(feature_array) @ s_array @ feature_array
    regulariser = lambda_ * np.eye(feature_array.shape[1])
    cost_matrix = np.linalg.solve(symmetric_matrix + regulariser, np.transpose(feature_array))
    return feature_array @ cost_matrix @ s_array


def leave_k_out_predict(target_array, hat_matrix, removal_count):
    """
    Compute the leave-k-out residuals of the model's features.
    :param target_array: The model's targets.
    :param hat_matrix: The model's weighted least squares Hessian matrix.
    :param removal_count: The amount of data points to be removed.
    :return: The predicted labels for the removed data points.
    """
    leave_one_out = np.zeros(removal_count)
    for i in range(removal_count):
        leave_one_out[i] = (target_array[i] - hat_matrix[i] @ target_array) / (1 - hat_matrix[i, i])

    s_matrix = np.eye(removal_count)
    for i in range(removal_count):
        for j in range(removal_count):
            if j != i:
                s_matrix[i, j] = -hat_matrix[i, j] / (1 - hat_matrix[i, j])

    leave_k_out = np.linalg.solve(s_matrix, leave_one_out)
    return target_array[:removal_count] - np.expand_dims(leave_k_out, 1)


def gram_schmidt(matrix):
    """
    Perform the Gram-Schmidt process on a matrix.
    :param matrix: The target matrix.
    :return: The Gram-Schmidt decomposition of the matrix.
    """
    k, d = matrix.shape
    if k <= d:
        q, r = np.linalg.qr(np.transpose(matrix))
    else:
        q, r = np.linalg.qr(np.transpose(matrix), mode="complete")
    u = np.transpose(q)
    c = np.transpose(r)
    return u, c


def projective_residual_update(weight_array, feature_array, target_array, hat_matrix, removal_count):
    """
    Perform the projective residual update procedure on linear regression models.
    :param weight_array: The model's weights.
    :param feature_array: The model's features.
    :param target_array: The model's targets.
    :param hat_matrix: The 'hat' matrix.
    :param removal_count: The amount of data points to be removed.
    :return: The updated model weights after projective residual update.
    """
    lko_prediction = leave_k_out_predict(target_array, hat_matrix, removal_count)

    u, c = gram_schmidt(feature_array[:removal_count])
    c_matrix = np.transpose(c) @ c
    eigenvalues, eigenvectors = np.linalg.eigh(c_matrix)
    v_matrix = np.transpose(eigenvectors) @ u

    gradient = np.zeros(feature_array.shape[1])
    for i in range(removal_count):
        gradient += (np.dot(feature_array[i], np.transpose(weight_array)) - lko_prediction[i]) * feature_array[i]

    step = np.zeros(feature_array.shape[1])
    for i in range(removal_count):
        factor = 0
        if eigenvalues[i] > 1e-10:
            factor = 1 / eigenvalues[i]
        step += factor * np.dot(v_matrix[i], gradient) * v_matrix[i]

    updated_weights = weight_array - step
    return updated_weights


def logistic_pru(removal_count, feature_array, target_array, weight_array, lambda_=LAMBDA, input_hessian=None):
    """
    Perform the projective residual update procedure on logistic regression models.
    :param removal_count: The amount of data points to be removed.
    :param feature_array: The model's features.
    :param target_array: The model's targets
    :param weight_array: The model's weights
    :param lambda_: The regularization parameter.
    :param input_hessian: The model's precomputed Hessian matrix (if exists).
    :return: The updated model weights after projective residual update.
    """
    logits = scipy.special.expit(feature_array @ np.transpose(weight_array))
    w_matrix = logits * (1 - logits)
    s_matrix = np.diagflat(w_matrix)
    z_matrix = feature_array @ np.transpose(weight_array) + np.linalg.inv(s_matrix) @ (
        np.expand_dims(target_array, 1) - logits
    )
    if input_hessian is None:
        hessian_matrix = hessian(feature_array, s_matrix, lambda_)
    else:
        hessian_matrix = input_hessian

    return projective_residual_update(weight_array, feature_array, z_matrix, hessian_matrix, removal_count)


def calculate_hessian(feature_array, weight_array, lambda_=LAMBDA):
    """
    Calculate the Hessian matrix of a given model.
    :param feature_array: The model's features.
    :param weight_array: The model's weights.
    :param lambda_: The regularization parameter.
    :return: The model's Hessian matrix.
    """
    logits = scipy.special.expit(feature_array @ np.transpose(weight_array))
    w_matrix = logits * (1 - logits)
    s_matrix = np.diagflat(w_matrix)
    return hessian(feature_array, s_matrix, lambda_)


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

    weights_after_removal = logistic_pru(weights, feature, target, k)
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
