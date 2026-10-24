"""NumPy-only linear and binary logistic regression (no autograd)."""
from __future__ import annotations
import numpy as np


class LinearRegressionGD:
    def __init__(self, learning_rate: float = .03, epochs: int = 300, batch_size: int | None = None,
                 l2: float = 0., seed: int = 42):
        self.learning_rate, self.epochs, self.batch_size, self.l2, self.seed = learning_rate, epochs, batch_size, l2, seed
        self.weights: np.ndarray | None = None
        self.bias = 0.
        self.history: list[float] = []

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.weights is None: raise RuntimeError("Model has not been fitted")
        return np.asarray(X) @ self.weights + self.bias

    @staticmethod
    def loss_and_gradients(X: np.ndarray, y: np.ndarray, weights: np.ndarray, bias: float, l2: float = 0.):
        n = len(X); error = X @ weights + bias - y.reshape(-1)
        loss = float(np.mean(error ** 2) / 2 + l2 * np.sum(weights ** 2) / 2)
        return loss, X.T @ error / n + l2 * weights, float(np.mean(error))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegressionGD":
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float).reshape(-1)
        if X.ndim != 2 or len(X) != len(y) or not len(X): raise ValueError("Expected nonempty 2D X and matching y")
        rng = np.random.default_rng(self.seed); self.weights = np.zeros(X.shape[1]); self.bias = 0.; self.history = []
        batch = self.batch_size or len(X)
        for _ in range(self.epochs):
            for ids in (rng.permutation(len(X))[i:i+batch] for i in range(0, len(X), batch)):
                _, dw, db = self.loss_and_gradients(X[ids], y[ids], self.weights, self.bias, self.l2)
                self.weights -= self.learning_rate * dw; self.bias -= self.learning_rate * db
            self.history.append(float(np.mean((self.predict(X)-y)**2)))
        return self

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        y = np.asarray(y).reshape(-1); p = self.predict(X)
        return float(1 - np.sum((y-p)**2) / max(np.sum((y-y.mean())**2), 1e-15))


class LogisticRegressionGD:
    def __init__(self, learning_rate: float = .1, epochs: int = 300, batch_size: int = 32, l2: float = .01, seed: int = 42):
        self.learning_rate, self.epochs, self.batch_size, self.l2, self.seed = learning_rate, epochs, batch_size, l2, seed
        self.weights: np.ndarray | None = None; self.bias = 0.; self.history: list[float] = []

    @staticmethod
    def sigmoid(z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=float); out = np.empty_like(z); pos = z >= 0
        out[pos] = 1 / (1 + np.exp(-z[pos])); ez = np.exp(z[~pos]); out[~pos] = ez / (1 + ez)
        return out

    @staticmethod
    def loss_and_gradients(X: np.ndarray, y: np.ndarray, weights: np.ndarray, bias: float, l2: float = 0.):
        y = y.reshape(-1); p = LogisticRegressionGD.sigmoid(X @ weights + bias)
        eps = np.finfo(float).eps
        loss = float(-np.mean(y*np.log(np.clip(p,eps,1)) + (1-y)*np.log(np.clip(1-p,eps,1))) + l2*np.sum(weights**2)/2)
        err = p-y
        return loss, X.T @ err / len(y) + l2*weights, float(err.mean())

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegressionGD":
        X, y = np.asarray(X,dtype=float), np.asarray(y,dtype=float).reshape(-1)
        if X.ndim != 2 or len(X)!=len(y) or not len(X) or not np.isin(y,[0,1]).all(): raise ValueError("Expected matching features and binary labels")
        rng=np.random.default_rng(self.seed); self.weights=np.zeros(X.shape[1]); self.bias=0.; self.history=[]
        for _ in range(self.epochs):
            order=rng.permutation(len(X))
            for start in range(0,len(X),self.batch_size):
                ix=order[start:start+self.batch_size]; _,dw,db=self.loss_and_gradients(X[ix],y[ix],self.weights,self.bias,self.l2)
                self.weights-=self.learning_rate*dw; self.bias-=self.learning_rate*db
            self.history.append(self.loss_and_gradients(X,y,self.weights,self.bias,self.l2)[0])
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.weights is None: raise RuntimeError("Model has not been fitted")
        return self.sigmoid(np.asarray(X)@self.weights+self.bias)

    def predict(self, X: np.ndarray, threshold: float=.5) -> np.ndarray: return (self.predict_proba(X)>=threshold).astype(int)

    def score(self,X:np.ndarray,y:np.ndarray)->float: return float(np.mean(self.predict(X)==np.asarray(y).reshape(-1)))

    def metrics(self, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
        truth=np.asarray(y).reshape(-1); pred=self.predict(X)
        tp=float(np.sum((truth==1)&(pred==1))); fp=float(np.sum((truth==0)&(pred==1))); fn=float(np.sum((truth==1)&(pred==0)))
        precision=tp/max(tp+fp,1.); recall=tp/max(tp+fn,1.)
        return {"accuracy":float(np.mean(pred==truth)),"precision":precision,"recall":recall,"f1":2*precision*recall/max(precision+recall,1e-15)}
