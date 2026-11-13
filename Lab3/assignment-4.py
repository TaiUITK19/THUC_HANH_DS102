import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression as SklearnLinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from linear_regression import LinearRegression

df = pd.read_csv("../Lab1/openalex_papers.csv")

x = df[["Năm xuất bản", "Số tác giả"]].to_numpy(dtype=np.float64)
y = df["Số lượt trích dẫn"].to_numpy(dtype=np.float64)

np.random.seed(42)
indices = np.random.permutation(len(x))

train_size = int(0.8 * len(x))

train_idx = indices[:train_size]
test_idx = indices[train_size:]

x_train = x[train_idx]
y_train = y[train_idx]

x_test = x[test_idx]
y_test = y[test_idx]

#Mô hình tự cài đặt
model = LinearRegression()
model.fit(x_train, y_train)
y_hat = model.predict(x_test)
results = model.evaluate(y_test, y_hat)


print("Kết quả đánh giá mô hình tự cài đặt:")
print("Hệ số hồi quy: ", model.beta)
print("Giá trị hàm mất mát: ", model.losses[-1])
print ("Kết quả đánh giá:")
for metric, value in results.items():
    print(f"{metric}: {value:.4f}")

#Mô hình Scikit-Learn
sk = SklearnLinearRegression()
sk.fit(x_train, y_train)
y_hat_sk = sk.predict(x_test)

results_sk = {
        "mse": mean_squared_error(y_test, y_hat_sk),
        "rmse": np.sqrt(mean_squared_error(y_test, y_hat_sk)),
        "mae": mean_absolute_error(y_test, y_hat_sk),
        "r2 score": r2_score(y_test, y_hat_sk)
}


print("\nKết quả đánh giá mô hình Scikit-Learn:")
print("Hệ số hồi quy: ", sk.coef_)
print ("Kết quả đánh giá:")
for metric, value in results_sk.items():
    print(f"{metric}: {value:.4f}")
