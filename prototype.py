import numpy as np
import scipy.sparse as sp
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import matplotlib.pyplot as plt
import os

# ------------------------
# Step 1: Dataset
# ------------------------
data = load_iris()
X = data.data
y = data.target.reshape(-1, 1)

scaler = StandardScaler()
X = scaler.fit_transform(X)

enc = OneHotEncoder(sparse_output=False)
y = enc.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------------
# Step 2: Init Sparse Weights
# ------------------------
def init_sparse(shape, sparsity=0.5):
    dense = np.random.randn(*shape) * 0.01
    mask = np.random.rand(*shape) > sparsity
    dense = dense * mask
    return sp.csr_matrix(dense)

input_dim = X_train.shape[1]
hidden_dim = 64
output_dim = y_train.shape[1]

W1 = init_sparse((input_dim, hidden_dim), sparsity=0.5)
W2 = init_sparse((hidden_dim, output_dim), sparsity=0.5)

# ------------------------
# Step 3: Forward + Activations
# ------------------------
def relu(x): return np.maximum(0, x)

def softmax(x):
    e = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e / np.sum(e, axis=1, keepdims=True)

def forward(X, W1, W2):
    H = relu(X @ W1.toarray())
    out = softmax(H @ W2.toarray())
    return H, out

# ------------------------
# Step 4: Loss + Backprop
# ------------------------
def cross_entropy(y_pred, y_true):
    return -np.mean(np.sum(y_true * np.log(y_pred + 1e-9), axis=1))

def accuracy(y_pred, y_true):
    return np.mean(np.argmax(y_pred, axis=1) == np.argmax(y_true, axis=1))

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

    momentum_W1 = beta * momentum_W1 + (1 - beta) * dW1
    momentum_W2 = beta * momentum_W2 + (1 - beta) * dW2

    W1 = sp.csr_matrix(W1.toarray() - lr * momentum_W1)
    W2 = sp.csr_matrix(W2.toarray() - lr * momentum_W2)
    return W1, W2

# ------------------------
# Step 5: Growth + Pruning
# ------------------------
def adjust_network(W, threshold=1e-3, growth_prob=0.05):
    arr = W.toarray()
    before_nnz = np.count_nonzero(arr)

    # pruning
    pruned_mask = np.abs(arr) < threshold
    arr[pruned_mask] = 0
    pruned_count = np.sum(pruned_mask)

    # growth
    growth_mask = (arr == 0) & (np.random.rand(*arr.shape) < growth_prob)
    arr[growth_mask] = np.random.randn(np.sum(growth_mask)) * 0.01
    growth_count = np.sum(growth_mask)

    after_nnz = np.count_nonzero(arr)

    return sp.csr_matrix(arr), growth_count, pruned_count, after_nnz

# ------------------------
# Logging helpers
# ------------------------
def weight_stats(W):
    arr = W.toarray()
    nonzeros = arr[arr != 0]
    if nonzeros.size == 0:
        return 0.0, 0.0
    return float(np.mean(np.abs(nonzeros))), float(np.std(nonzeros))

def estimate_flops(batch_size, nnz_total):
    return 2 * batch_size * nnz_total * 3  # fwd+back ~3x

def save_mask(W, path):
    mask = (W.toarray() != 0).astype(int)
    plt.imshow(mask, aspect="auto")
    plt.title(path)
    plt.savefig(path)
    plt.close()

# ------------------------
# Step 6: Training Loop
# ------------------------
epochs = 50
batch_size = 16

OUT_DIR = "mosaiс_logs"
os.makedirs(OUT_DIR, exist_ok=True)

epoch_stats = {"epoch": [], "nnz_W1": [], "nnz_W2": [],
               "growth": [], "prune": [], "flops": [],
               "mean_abs_W1": [], "mean_abs_W2": []}

for epoch in range(epochs):
    perm = np.random.permutation(X_train.shape[0])
    X_train_shuffled, y_train_shuffled = X_train[perm], y_train[perm]

    growth_events, prune_events, flops_total = 0, 0, 0

    for i in range(0, X_train.shape[0], batch_size):
        X_batch = X_train_shuffled[i:i+batch_size]
        y_batch = y_train_shuffled[i:i+batch_size]

        H, out = forward(X_batch, W1, W2)
        W1, W2 = backprop(X_batch, H, y_batch, out, W1, W2)

        W1, g1, p1, nnz_W1 = adjust_network(W1)
        W2, g2, p2, nnz_W2 = adjust_network(W2)
        growth_events += g1 + g2
        prune_events += p1 + p2

        flops_total += estimate_flops(X_batch.shape[0], nnz_W1 + nnz_W2)

    H_full, out_full = forward(X_train, W1, W2)
    loss = cross_entropy(out_full, y_train)
    acc = accuracy(out_full, y_train)

    # log
    epoch_stats["epoch"].append(epoch)
    epoch_stats["nnz_W1"].append(nnz_W1)
    epoch_stats["nnz_W2"].append(nnz_W2)
    epoch_stats["growth"].append(growth_events)
    epoch_stats["prune"].append(prune_events)
    epoch_stats["flops"].append(flops_total)
    mean1, _ = weight_stats(W1)
    mean2, _ = weight_stats(W2)
    epoch_stats["mean_abs_W1"].append(mean1)
    epoch_stats["mean_abs_W2"].append(mean2)

    if epoch % 10 == 0:
        save_mask(W1, os.path.join(OUT_DIR, f"W1_mask_epoch{epoch}.png"))

    print(f"Epoch {epoch+1}/{epochs} | Loss: {loss:.4f} | Acc: {acc:.4f} "
          f"| Growth: {growth_events} | Prune: {prune_events}")

# ------------------------
# Step 7: Evaluation
# ------------------------
_, out_test = forward(X_test, W1, W2)
print("Final Test Accuracy:", accuracy(out_test, y_test))
print("Final W1 active connections:", W1.count_nonzero())
print("Final W2 active connections:", W2.count_nonzero())

# ------------------------
# Step 8: Plots
# ------------------------
epochs_range = epoch_stats["epoch"]

plt.plot(epochs_range, epoch_stats["nnz_W1"], label="nnz W1")
plt.plot(epochs_range, epoch_stats["nnz_W2"], label="nnz W2")
plt.xlabel("Epoch"); plt.ylabel("Active connections"); plt.legend()
plt.title("Active connections over epochs")
plt.savefig(os.path.join(OUT_DIR, "nnz_over_epochs.png")); plt.close()

plt.plot(epochs_range, epoch_stats["growth"], label="growth")
plt.plot(epochs_range, epoch_stats["prune"], label="prune")
plt.xlabel("Epoch"); plt.ylabel("Events"); plt.legend()
plt.title("Growth vs Prune events")
plt.savefig(os.path.join(OUT_DIR, "growth_prune_events.png")); plt.close()

plt.plot(epochs_range, epoch_stats["flops"])
plt.xlabel("Epoch"); plt.ylabel("FLOPs per epoch")
plt.title("FLOPs over epochs")
plt.savefig(os.path.join(OUT_DIR, "flops_over_epochs.png")); plt.close()

plt.plot(epochs_range, epoch_stats["mean_abs_W1"], label="mean_abs_W1")
