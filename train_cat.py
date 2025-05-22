# train_cat.py
import pandas as pd
import joblib
import os
from catboost import CatBoostRegressor

# 加载训练集
X_train = pd.read_csv("prepare/X_train.csv")
y_train = pd.read_csv("prepare/y_train.csv").squeeze()

# 模型训练
model = CatBoostRegressor(verbose=0, random_state=42)
model.fit(X_train, y_train)

# 保存模型
os.makedirs("models", exist_ok=True)
model.save_model("models/cat_model.cbm")

print("✅ CatBoost 模型训练完成，已保存为 models/cat_model.cbm")
