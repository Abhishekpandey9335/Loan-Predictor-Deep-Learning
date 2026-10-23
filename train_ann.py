import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, roc_auc_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# 1. Load data
df = pd.read_csv("loan_default_100k.csv")   # same folder as this script
X = df.drop(columns=["default"])
y = df["default"].values

num_cols = X.select_dtypes(include="number").columns.tolist()
cat_cols = X.select_dtypes(exclude="number").columns.tolist()

# 2. Preprocessing: impute + scale numbers, one-hot encode categories
pre = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("sc", StandardScaler())]), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

X_train = pre.fit_transform(X_train)
X_test = pre.transform(X_test)
if hasattr(X_train, "toarray"):
    X_train, X_test = X_train.toarray(), X_test.toarray()

# 3. ANN model
model = keras.Sequential([
    layers.Input(shape=(X_train.shape[1],)),
    layers.Dense(64, activation="relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(32, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(16, activation="relu"),
    layers.Dense(1, activation="sigmoid"),
])
model.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="binary_crossentropy",
              metrics=["accuracy", keras.metrics.AUC(name="auc")])
model.summary()

# 4. Train
early = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
history = model.fit(X_train, y_train, validation_split=0.2,
                    epochs=50, batch_size=256, callbacks=[early], verbose=2)

# 5. Evaluate
proba = model.predict(X_test).ravel()
pred = (proba > 0.5).astype(int)
print(classification_report(y_test, pred))
print("ROC-AUC:", roc_auc_score(y_test, proba))

model.save("loan_ann_model.keras")

import joblib
joblib.dump(pre, 'preprocessor.joblib')
