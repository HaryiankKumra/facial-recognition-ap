
import numpy as np

class SimpleELMClassifier:
    def __init__(self, n_hidden=300, activation='relu', random_state=42):
        self.n_hidden = n_hidden
        self.activation = activation
        self.random_state = random_state
        self.is_fitted = False

    def _activation(self, X):
        if self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-X))
        elif self.activation == 'tanh':
            return np.tanh(X)
        elif self.activation == 'relu':
            return np.maximum(0, X)
        else:
            raise ValueError('Unsupported activation')

    def fit(self, X, y):
        np.random.seed(self.random_state)
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        self.classes_ = np.unique(y)
        Y = np.zeros((n_samples, n_classes))
        for i, label in enumerate(self.classes_):
            Y[y == label, i] = 1

        self.W = np.random.randn(n_features, self.n_hidden)
        self.b = np.random.randn(self.n_hidden)
        H = self._activation(np.dot(X, self.W) + self.b)
        self.beta = np.dot(np.linalg.pinv(H), Y)
        self.is_fitted = True

    def predict_proba(self, X):
        if not self.is_fitted:
            raise Exception("Model not fitted yet.")
        H = self._activation(np.dot(X, self.W) + self.b)
        logits = np.dot(H, self.beta)
        e_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return e_logits / np.sum(e_logits, axis=1, keepdims=True)

    def predict(self, X):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)
