# prepare/normalize.py
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import os

# 加载数据
X_train = pd.read_csv("prepare/X_train.csv")
X_val = pd.read_csv("prepare/X_val.csv")
X_full = pd.read_csv("prepare/X_full.csv")

# 创建标准化器
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_full_scaled = scaler.transform(X_full)

# 保存标准化后的数据
pd.DataFrame(X_train_scaled, columns=X_train.columns).to_csv("prepare/X_train_scaled.csv", index=False)
pd.DataFrame(X_val_scaled, columns=X_val.columns).to_csv("prepare/X_val_scaled.csv", index=False)
pd.DataFrame(X_full_scaled, columns=X_full.columns).to_csv("prepare/X_full_scaled.csv", index=False)

# 保存 scaler 模型
os.makedirs("models", exist_ok=True)
joblib.dump(scaler, "models/mlp_scaler.joblib")
print("✅ 数据标准化完成，保存为 X_train_scaled.csv 等")
