import numpy as np

class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.loss_history = []

    def fit(self, X, y):
         n_samples, n_features = X.shape

         self.weights = np.zeros(n_features)
         self.bias = 0

         for i in range(self.n_iterations):
             y_pred = np.dot(X, self.weights) + self.bias

             dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
             db = (1 / n_samples) * np.sum(y_pred - y)

             self.weights -= self.learning_rate * dw
             self.bias -= self.learning_rate * db

             loss = np.mean((y_pred - y)**2)
             self.loss_history.append(loss)

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias
    
    def get_params(self):
        return {'weights': self.weights, 'bias': self.bias}