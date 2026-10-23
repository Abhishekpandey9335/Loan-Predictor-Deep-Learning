import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score

# 1. Device (GPU mile toh GPU, warna CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device, torch.cuda.get_device_name(0) if device.type == "cuda" else "")

# 2. Data load
df = pd.read_csv("loan_default_100k.csv")
X = df.drop(columns=["default"])
y = df["default"].values.astype("float32")

num_cols = X.select_dtypes(include="number").columns.tolist()
cat_cols = X.select_dtypes(exclude="number").columns.tolist()

pre = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("sc", StandardScaler())]), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

X_train = pre.fit_transform(X_train).astype("float32")
X_val = pre.transform(X_val).astype("float32")
X_test = pre.transform(X_test).astype("float32")


def to_tensor(a):
    return torch.tensor(a, dtype=torch.float32)


train_dl = DataLoader(TensorDataset(to_tensor(X_train), to_tensor(y_train)),
                      batch_size=256, shuffle=True)
X_val_t, y_val_t = to_tensor(X_val).to(device), to_tensor(y_val).to(device)

# 3. Model (GPU pe bhejo)
model = nn.Sequential(
    nn.Linear(X_train.shape[1], 64), nn.ReLU(), nn.BatchNorm1d(64), nn.Dropout(0.3),
    nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.2),
    nn.Linear(32, 16), nn.ReLU(),
    nn.Linear(16, 1),
).to(device)

loss_fn = nn.BCEWithLogitsLoss()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

# 4. Train (early stopping ke saath)
best_val, best_state, patience, wait = float("inf"), None, 5, 0
for epoch in range(50):
    model.train()
    for xb, yb in train_dl:
        xb, yb = xb.to(device), yb.to(device)  # batch GPU pe
        opt.zero_grad()
        loss = loss_fn(model(xb).squeeze(1), yb)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        val_loss = loss_fn(model(X_val_t).squeeze(1), y_val_t).item()
    print(f"Epoch {epoch + 1:02d} | val_loss: {val_loss:.4f}")

    if val_loss < best_val:
        best_val, wait = val_loss, 0
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
    else:
        wait += 1
        if wait >= patience:
            print("Early stopping")
            break

model.load_state_dict(best_state)

# 5. Evaluate
model.eval()
with torch.no_grad():
    logits = model(to_tensor(X_test).to(device)).squeeze(1)
    proba = torch.sigmoid(logits).cpu().numpy()
pred = (proba > 0.5).astype(int)
print(classification_report(y_test, pred))
print("ROC-AUC:", roc_auc_score(y_test, proba))

torch.save(model.state_dict(), "loan_ann_model.pt")
