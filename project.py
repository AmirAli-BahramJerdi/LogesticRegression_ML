import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

df = load_digits()
X = df.data[:,8:]
y = df.target

mu = X.mean(axis=0)
s = np.std(X,axis=0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=45)

sigmoid = lambda z : 1/(1+np.exp(-z))


def fit_regression_logistic(X, y):
    W = []
    
    tolerance = 1e-4
    m = len(X)
    X = np.c_[np.ones(m)]
    
    for c in np.unique(y):
        y1 = pd.factorize(y==c,sort=True)[0].reshape(-1,1)
        alpha = 0.005
        w = np.zeros((X.shape[1],1))

        for _ in range(1000000):
            y_hat = sigmoid(X.dot(w))

            grad = X.T.dot(y_hat-y1)
            w -= alpha*grad/m

            if np.abs((alpha*grad).mean())<=tolerance:
                break
        W.append(w)
    return W

def predict(X):
    X = np.c_[np.ones(X.shape[0])]
    return np.argmax(sigmoid(X.dot(W)), axis=1)

W = fit_regression_logistic(X_train, y_train)

print('Accuracy train =',accuracy_score(y_train, predict(X_train)))
print('Accuracy test  =',accuracy_score(y_test, predict(X_test)))
print('Confiusion matrix :\n',confusion_matrix(y, predict(X)))