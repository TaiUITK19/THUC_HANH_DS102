import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from logistic_regression import LogisticsRegression

df = pd.read_csv("../Lab1/openalex_papers.csv")


texts = df["Tiêu đề"].to_numpy()
labels = df["Nhãn"].to_numpy()

train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=0.2, stratify=labels, random_state=42
)

train_data = (train_texts, train_labels)
test_data = (test_texts, test_labels)


class_1 = "Computer Science"

train_labels = (train_labels == class_1).astype(np.float64)
test_labels = (test_labels == class_1).astype(np.float64)

vectorizer = TfidfVectorizer(
        ngram_range=(1, 2), min_df=2, sublinear_tf=True, stop_words="english"
)

train_texts = vectorizer.fit_transform(train_texts).toarray()
test_texts = vectorizer.transform(test_texts).toarray()

model = LogisticsRegression(
        epoch=1000,
        lr=5
)

model.fit(train_texts, train_labels)
predicted_labels = model.predict(test_texts)
metrics = model.evaluate(test_labels, predicted_labels)
print("\n\n")
for metric in metrics:
        print(f"{metric}: {metrics[metric]}")

plt.figure(figsize=(8,5))
plt.plot(range(1, len(model.losses) + 1), model.losses, color="tab:blue", linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("Cross-entropy loss")
plt.title("Logistic Regression - Training Loss")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()