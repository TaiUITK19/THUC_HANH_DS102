import numpy as np
import pandas as pd

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

# HUẤN LUYỆN VỚI DỮ LIỆU GÔCS
model_raw = LinearRegression()
model_raw.fit(x_train, y_train)
y_hat_raw = model_raw.predict(x_test)
results_raw = model_raw.evaluate(y_test, y_hat_raw)

# HUẤN LUYỆN VỚI DỮ LIỆU ĐÃ CHUẨN HÓA
mean = x_train.mean(axis=0)
std = x_train.std(axis=0)
std[std == 0] = 1

x_train = (x_train - mean) / std
x_test = (x_test - mean) / std

model = LinearRegression()
model.fit(x_train, y_train)

y_hat = model.predict(x_test)

results = model.evaluate(y_test, y_hat)

print("Kết quả đánh giá với dữ liệu gốc:")
print("Hệ số hồi quy: ", model_raw.beta)
print("Giá trị hàm mất mát: ", model_raw.losses[-1])
print ("Kết quả đánh giá:")
for metric, value in results_raw.items():
    print(f"{metric}: {value:.4f}")


print("Kết quả đánh giá với dữ liệu đã chuẩn hóa:")
print("Hệ số hồi quy: ", model.beta)
print("Giá trị hàm mất mát: ", model.losses[-1])

print ("Kết quả đánh giá:")
for metric, value in results.items():
    print(f"{metric}: {value:.4f}")



