import numpy as np
class LinearRegression:
        
        def __init__(self):
                self.beta = None
                self.losses = []

        
        def loss_fn(self, y: np.ndarray, y_hat: np.ndarray) -> float:
                loss = 0.5 * np.sum((y - y_hat) ** 2)
                return loss
        
        def fit(self, X: np.ndarray, y: np.ndarray):
                X = np.asarray(X, dtype=np.float64)
                y = np.asarray(y, dtype=np.float64).reshape(-1)
                N, k = X.shape

                Xb = np.c_[np.ones((N, 1)), X]

                try:
                        self.beta = np.linalg.solve(Xb.T @ Xb, Xb.T @ y)
                except np.linalg.LinAlgError:
                        self.beta = np.linalg.lstsq(Xb, y, rcond=None)[0]

                y_hat = Xb @ self.beta
                self.losses = [self.loss_fn(y, y_hat)]
                
                return self

        
        def predict(self, X: np.ndarray) -> np.ndarray:
                if self.beta is None:
                        raise ValueError("Hãy gọi fit() trước khi gọi predict().")
                X=np.asarray(X, dtype=np.float64)
                if X.ndim == 1:
                        X = X.reshape(1, -1)
                Xb = np.c_[np.ones((X.shape[0], 1)), X]
                y_hat = Xb @ self.beta

                return y_hat
        
        def evaluate(self, y: np.ndarray, y_hat: np.ndarray) -> dict:
                y = np.asarray(y, dtype=np.float64).reshape(-1)
                y_hat = np.asarray(y_hat, dtype=np.float64).reshape(-1)
                mse = np.mean((y - y_hat) ** 2)
                rmse = np.sqrt(mse)
                mae = np.mean(np.abs(y - y_hat))
                ss_res = np.sum((y - y_hat) ** 2)
                ss_total = np.sum((y - np.mean(y)) ** 2)
                r2 = 1 - (ss_res / ss_total) if ss_total != 0 else float('nan')  

                return {
                        "mse": mse,
                        "rmse": rmse,
                        "mae": mae,
                        "r2 score": r2
                }
        