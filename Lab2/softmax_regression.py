import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm

class SoftmaxRegression:
        def __init__(self, epoch: int, lr: float):
                self.epoch = epoch
                self.lr = lr
                self.w = None
                self.losses = []

        def loss_fn(self, y: np.ndarray, y_hat: np.ndarray):
                return -(y * np.log(y_hat + 10e-15)).sum(axis=-1).mean()

        def fit(self, X: np.ndarray, y: np.ndarray):
                N, d = X.shape
                _, k = y.shape
                self.w = np.zeros((d, k))
                e = 0
                with tqdm(range(self.epoch)) as pbar:
                        for _ in pbar:
                                # forward pass
                                y_hat = self.predict(X)
                                
                                #backward pass
                                delta_y = (y_hat - y) #(N,k)
                                gradient = delta_y.T @ X #(k,d)
                                self.w = self.w - (1/N) * self.lr * gradient.T #(d,k)
                                
                                l = self.loss_fn(y, y_hat)
                                self.losses.append(l)
                                
                                pbar.set_postfix({
                                        "Epoch": e+1,
                                        "loss": l
                                })
                                e += 1

        def softmax(self, z: np.ndarray):
                z = z - z.max(axis=-1, keepdims=True )
                denum = np.exp(z).sum(axis=-1, keepdims=True)
                return np.exp(z) / denum

        def predict(self, X: np.ndarray) -> np.ndarray:
                z = X @ self.w 
                y_hat = self.softmax(z)
                return y_hat

        def evaluate(self, y, y_hat) -> dict:
                y = np.asarray(y)
                if y.ndim == 2:
                        y = np.argmax(y, axis=-1)
                y_hat = np.argmax(y_hat, axis=-1)
                precision = precision_score(y, y_hat, average="macro", zero_division= 0)
                recall = recall_score(y, y_hat, average="macro", zero_division= 0)
                f1 = f1_score(y, y_hat, average="macro", zero_division= 0)
                
                return {
                        "precision": precision,
                        "recall": recall,
                        "f1": f1
                }