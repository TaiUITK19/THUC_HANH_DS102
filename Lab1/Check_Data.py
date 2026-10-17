import pandas as pd

df = pd.read_csv("openalex_papers.csv")

print("Số dòng x số cột: ", df.shape)

print("\n Số giá trị thiếu mỗi cột:")
thieu = df.isna() .sum()
print(pd.DataFrame({"Số dòng thiếu": thieu}))

print("\nSố dòng trùng lặp", df.duplicated().sum())

print("\Kiểm tra các cột số:")
print(df[["Năm xuất bản", "Số lượt trích dẫn", "Số tác giả"]].agg(["min", "max", "mean", "std"]))

print("\Số bài mỗi nhãn:")
print(df["Nhãn"].value_counts())