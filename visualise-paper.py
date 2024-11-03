import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from scipy.interpolate import make_interp_spline

sns.set_theme()
sns.color_palette("colorblind")

test_size = (0.2, 0.3, 0.4)
lambda_ = (0.001, 0.005, 0.01)
sample = (16, 14, 12)

removal_count = (np.arange(1, 17), np.arange(1, 15), np.arange(1, 13))
removal_count_interpolated = (np.arange(1, 16, 0.01), np.arange(1, 14, 0.01), np.arange(1, 12, 0.01))
x_ticks = np.arange(1, 11)

l2_distance = joblib.load("out/rev/l2_distance-latest.joblib")
accuracy = joblib.load("out/rev/accuracy-latest.joblib")
runtime = joblib.load("out/runtime.joblib")
runtime_pru_result = joblib.load("out/runtime_pru.joblib")

l2_distance_base = np.median(l2_distance["base"], axis=3)
l2_distance_cr = np.median(l2_distance["cr"], axis=3)
l2_distance_pru = np.median(l2_distance["pru"], axis=3)

accuracy_naive = np.median(accuracy["naive"], axis=3)
accuracy_cr = np.median(accuracy["cr"], axis=3)
accuracy_pru = np.median(accuracy["pru"], axis=3)

runtime_naive = np.median(runtime["naive"], axis=1)
runtime_cr = np.median(runtime["cr"], axis=1)
runtime_pru = np.median(runtime_pru_result["pru"], axis=1)

l2_distance_cr_interpolated_all = []
l2_distance_pru_interpolated_all = []
accuracy_cr_interpolated_all = []
accuracy_pru_interpolated_all = []

for i in range(len(test_size)):
    l2_distance_base_interpolated = []
    l2_distance_cr_interpolated = []
    l2_distance_pru_interpolated = []
    accuracy_naive_interpolated = []
    accuracy_cr_interpolated = []
    accuracy_pru_interpolated = []
    runtime_naive_interpolated = []
    runtime_cr_interpolated = []
    runtime_pru_interpolated = []
    for j in range(len(lambda_)):
        l2_distance_base_function = make_interp_spline(removal_count[i], l2_distance_base[i][j][: sample[i]], k=1)
        l2_distance_base_interpolated.append(l2_distance_base_function(removal_count_interpolated[i]))

        l2_distance_cr_function = make_interp_spline(removal_count[i], l2_distance_cr[i][j][: sample[i]], k=1)
        l2_distance_cr_interpolated.append(l2_distance_cr_function(removal_count_interpolated[i]))
        l2_distance_cr_interpolated_all.append(l2_distance_cr_function(removal_count_interpolated[i]))

        l2_distance_pru_function = make_interp_spline(removal_count[i], l2_distance_pru[i][j][: sample[i]], k=1)
        l2_distance_pru_interpolated.append(l2_distance_pru_function(removal_count_interpolated[i]))
        l2_distance_pru_interpolated_all.append(l2_distance_pru_function(removal_count_interpolated[i]))

        accuracy_naive_function = make_interp_spline(removal_count[i], accuracy_naive[i][j][: sample[i]], k=1)
        accuracy_naive_interpolated.append(accuracy_naive_function(removal_count_interpolated[i]))

        accuracy_cr_function = make_interp_spline(removal_count[i], accuracy_cr[i][j][: sample[i]], k=1)
        accuracy_cr_interpolated.append(accuracy_cr_function(removal_count_interpolated[i]))
        accuracy_cr_interpolated_all.append(accuracy_cr_function(removal_count_interpolated[i]))

        accuracy_pru_function = make_interp_spline(removal_count[i], accuracy_pru[i][j][: sample[i]], k=1)
        accuracy_pru_interpolated.append(accuracy_pru_function(removal_count_interpolated[i]))
        accuracy_pru_interpolated_all.append(accuracy_pru_function(removal_count_interpolated[i]))

    runtime_naive_function = make_interp_spline(removal_count[i], runtime_naive[: sample[i]], k=1)
    runtime_naive_interpolated = runtime_naive_function(removal_count_interpolated[i])

    runtime_cr_function = make_interp_spline(removal_count[i], runtime_cr[: sample[i]], k=1)
    runtime_cr_interpolated = runtime_cr_function(removal_count_interpolated[i])

    runtime_pru_function = make_interp_spline(removal_count[i], runtime_pru[: sample[i]], k=1)
    runtime_pru_interpolated = runtime_pru_function(removal_count_interpolated[i])

    plt.cla()
    plt.clf()
    plt.subplots()

    if i == 0:
        # plt.title("Trade-off between Proportion of Removals and Unlearn Time")
        plt.plot(removal_count_interpolated[i], runtime_cr_interpolated, label="CR")
        plt.plot(removal_count_interpolated[i], runtime_pru_interpolated, label="PRU")
        plt.plot(removal_count[i], runtime_cr[: sample[i]], "o")
        plt.plot(removal_count[i], runtime_pru[: sample[i]], "D")
        plt.xlabel("Proportion of Removals (%)")
        plt.ylabel(r"Unlearn Time ($s$)")
        plt.xticks(np.arange(sample[i] / 10, np.floor(sample[i] * 1.1), sample[i] / 10), labels=x_ticks)
        plt.legend()
        plt.tight_layout()
        plt.savefig("out/runtime_tradeoff_paper.svg")
        # plt.show()

plt.figure(figsize=(19.2, 4.8))
# plt.suptitle("Trade-off between Proportion of Removals and Accuracy")
for i in range(len(test_size)):
    plt.subplot(1, 3, i + 1)
    for j in range(len(lambda_)):
        plt.title(f"Test set ratio = ${test_size[i]}$")
        plt.plot(removal_count_interpolated[i], accuracy_cr_interpolated_all[3*i + j], label=f"CR, λ = {lambda_[j]}")
        plt.plot(removal_count_interpolated[i], accuracy_pru_interpolated_all[3*i + j], label=f"PRU, λ = {lambda_[j]}")
        plt.plot(removal_count[i], accuracy_cr[i][j][: sample[i]], "o")
        plt.plot(removal_count[i], accuracy_pru[i][j][: sample[i]], "D")
    plt.xlabel("Proportion of Removals (%)")
    plt.ylabel("Accuracy")
    plt.xticks(np.arange(sample[i] / 10, np.floor(sample[i] * 1.1), sample[i] / 10), labels=x_ticks)
    plt.legend()
plt.tight_layout()
plt.savefig("out/accuracy_tradeoff_paper.svg")
# plt.show()

plt.cla()
plt.clf()

plt.figure(figsize=(19.2, 4.8))
# plt.suptitle(f"Trade-off between Proportion of Removals and $L^2$-Distance")
for i in range(len(test_size)):
    plt.subplot(1, 3, i + 1)
    for j in range(len(lambda_)):
        plt.title(f"Test set ratio = ${test_size[i]}$")
        plt.plot(removal_count_interpolated[i], l2_distance_cr_interpolated_all[3*i + j], label=f"CR, λ = {lambda_[j]}")
        plt.plot(removal_count_interpolated[i], l2_distance_pru_interpolated_all[3*i + j], label=f"PRU, λ = {lambda_[j]}")
        plt.plot(removal_count[i], l2_distance_cr[i][j][: sample[i]], "o")
        plt.plot(removal_count[i], l2_distance_pru[i][j][: sample[i]], "D")
    plt.xlabel("Proportion of Removals (%)")
    plt.ylabel(r"$L^2$-Distance")
    plt.xticks(np.arange(sample[i] / 10, np.floor(sample[i] * 1.1), sample[i] / 10), labels=x_ticks)
    plt.legend()
plt.tight_layout()
plt.savefig(f"out/l2_distance_tradeoff_paper.svg")
# plt.show()

plt.cla()
plt.clf()
