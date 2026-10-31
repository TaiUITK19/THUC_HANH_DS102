import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder   
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score


df = pd.read_csv("../Lab1/openalex_papers.csv")


texts = df["Tiêu đề"].to_numpy()
labels = df["Nhãn"].to_numpy()

train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=0.2, stratify=labels, random_state=42
)

vectorizer = TfidfVectorizer(
        ngram_range=(1, 2), min_df=2, sublinear_tf=True, stop_words="english"
)

X_train = vectorizer.fit_transform(train_texts)
X_test = vectorizer.transform(test_texts)

#Logistic Regression
class_1 = "Computer Science"

y_train_bin = (train_labels == class_1).astype(np.float64)
y_test_bin = (test_labels == class_1).astype(np.float64)

model_bin = LogisticRegression(max_iter=1000)
model_bin.fit(X_train, y_train_bin)
pred_bin = model_bin.predict(X_test)

print("Logistic Regression - sklearn")
print(f"precision: {precision_score(y_test_bin, pred_bin)}")
print(f"recall: {recall_score(y_test_bin, pred_bin)}")
print(f"f1: {f1_score(y_test_bin, pred_bin)}")

#Sofmax Regression
encoder = LabelEncoder()
y_train = encoder.fit_transform(train_labels)
y_test = encoder.transform(test_labels)

model_multi = LogisticRegression(max_iter=1000)
model_multi.fit(X_train, y_train)
pred = model_multi.predict(X_test)

print("Softmax Regression - sklearn")
print(f"precision: {precision_score(y_test, pred,  average='macro', zero_division= 0)}")
print(f"recall: {recall_score(y_test, pred,  average='macro', zero_division= 0)}")
print(f"f1: {f1_score(y_test, pred,  average='macro', zero_division= 0)}")