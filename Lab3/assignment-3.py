import numpy as np
import pandas as pd

from linear_regression import LinearRegression

df = pd.read_csv("../Lab1/openalex_papers.csv")

x = df[["Năm xuất bản", "Số tác giả"]].to_numpy(dtype=np.float64)
y = df["Số lượt trích dẫn"].to_numpy(dtype=np.float64)

onehot = pd.get_dummies(df[["Loại tài liệu", "Nhãn"]], dtype=float )
x_ext = np.c_[x, onehot.to_numpy()]
names = ["Năm xuất bản", "Số tác giả"] + list(onehot.columns)

np.random.seed(42)
indices = np.random.permutation(len(x))

train_size = int(0.8 * len(x))

train_idx = indices[:train_size]
test_idx = indices[train_size:]

x_train = x[train_idx]
y_train = y[train_idx]

x_test = x[test_idx]
y_test = y[test_idx]

x_train_ext = x_ext[train_idx]
y_train_ext = y[train_idx]


# Loại cột phụ thuộc tuyến tính

def  remove_collinear(X):
    keep = []
    current = np.ones((X.shape[0], 1))
    for j in range(X.shape[1]):
        new_col = np.c_[current, X[:, j]]
        if np.linalg.matrix_rank(new_col) > current.shape[1]:
            keep.append(j)
            current = new_col
    return keep
keep = remove_collinear(x_train_ext)
x_new_train = x_train_ext[:, keep]
x_new_test = x_ext[test_idx][:, keep]


# HUẤN LUYỆN VỚI DỮ LIỆU GÔCS
model_raw = LinearRegression()
model_raw.fit(x_train, y_train)
y_hat_raw = model_raw.predict(x_test)
results_raw = model_raw.evaluate(y_test, y_hat_raw)

# HUẤN LUYỆN VỚI DỮ LIỆU ĐÃ XỬ LÝ

model = LinearRegression()
model.fit(x_new_train, y_train)
results = model.evaluate(y_test, model.predict(x_new_test))

print("Kết quả đánh giá với dữ liệu gốc:")
print("Hệ số hồi quy: ", model_raw.beta)
print("Giá trị hàm mất mát: ", model_raw.losses[-1])
print ("Kết quả đánh giá:")
for metric, value in results_raw.items():
    print(f"{metric}: {value:.4f}")


print("Kết quả đánh giá với dữ liệu đã trích chọn đặt trưng:")
print("Hệ số hồi quy: ", model.beta)
print("Giá trị hàm mất mát: ", model.losses[-1])

print ("Kết quả đánh giá:")
for metric, value in results.items():
    print(f"{metric}: {value:.4f}")



