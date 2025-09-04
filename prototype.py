import numpy as np
import scipy.sparse as sp
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Step 1: Load Dataset
data = load_iris()
X = data.data
y = data.target.reshape(-1,1)

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# One-hot encode labels
enc = OneHotEncoder(sparse_output=False)
y = enc.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Init Sparse Weights
def init_sparse(shape, sparsity=0.5):    
    dense = np.random.randn(*shape) * 0.01
    mask = np.random.rand(*shape) > sparsity
    dense = dense * mask
    return sp.csr_matrix(dense)

input_dim = X_train.shape[1]
hidden_dim = 64  # bigger hidden layer
output_dim = y_train.shape[1]

W1 = init_sparse((input_dim, hidden_dim), sparsity=0.5)
W2 = init_sparse((hidden_dim, output_dim), sparsity=0.5)

# Step 3: Forward Pass
def relu(x):
    return np.maximum(0, x)

def softmax(x):
    e = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e / np.sum(e, axis=1, keepdims=True)

def forward(X, W1, W2):
    H = relu(X @ W1.toarray())
    out = softmax(H @ W2.toarray())
    return H, out

# Step 4: Loss + Backprop
def cross_entropy(y_pred, y_true):
    return -np.mean(np.sum(y_true * np.log(y_pred + 1e-9), axis=1))

def accuracy(y_pred, y_true):
    return np.mean(np.argmax(y_pred, axis=1) == np.argmax(y_true, axis=1))

# Momentum for stability
momentum_W1 = np.zeros_like(W1.toarray())
momentum_W2 = np.zeros_like(W2.toarray())
beta = 0.9
lr = 0.05

def backprop(X, H, y, out, W1, W2, lr=0.05):
    global momentum_W1, momentum_W2
    m = X.shape[0]
    d_out = (out - y) / m
    dW2 = H.T @ d_out
    dH = d_out @ W2.toarray().T
    dH[H <= 0] = 0
    dW1 = X.T @ dH

    # Momentum update
    momentum_W1 = beta * momentum_W1 + (1 - beta) * dW1
    momentum_W2 = beta * momentum_W2 + (1 - beta) * dW2

    W1 = sp.csr_matrix(W1.toarray() - lr * momentum_W1)
    W2 = sp.csr_matrix(W2.toarray() - lr * momentum_W2)
    return W1, W2

# Step 5: Growth + Pruning
def adjust_network(W, threshold=1e-3, growth_prob=0.05):  # higher growth probability
    arr = W.toarray()
    arr[np.abs(arr) < threshold] = 0
    mask = (arr == 0) & (np.random.rand(*arr.shape) < growth_prob)
    arr[mask] = np.random.randn(np.sum(mask)) * 0.01
    return sp.csr_matrix(arr)

# Step 6: Training Loop
epochs = 100
batch_size = 16

for epoch in range(epochs):
    perm = np.random.permutation(X_train.shape[0])
    X_train_shuffled, y_train_shuffled = X_train[perm], y_train[perm]

    for i in range(0, X_train.shape[0], batch_size):
        X_batch = X_train_shuffled[i:i+batch_size]
        y_batch = y_train_shuffled[i:i+batch_size]
        H, out = forward(X_batch, W1, W2)
        W1, W2 = backprop(X_batch, H, y_batch, out, W1, W2)
        W1 = adjust_network(W1)
        W2 = adjust_network(W2)

    H_full, out_full = forward(X_train, W1, W2)
    loss = cross_entropy(out_full, y_train)
    acc = accuracy(out_full, y_train)
    print(f"Epoch {epoch+1}/{epochs} | Loss: {loss:.4f} | Acc: {acc:.4f}")

# Final Test Accuracy
_, out_test = forward(X_test, W1, W2)
print("Test Accuracy:", accuracy(out_test, y_test))

def count_nonzeros(W):
    return W.count_nonzero()

print("W1 active connections:", count_nonzeros(W1))
print("W2 active connections:", count_nonzeros(W2))
