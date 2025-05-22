# train_xgb.py
import pandas as pd
import os
import xgboost as xgb

# 加载训练集
X_train = pd.read_csv("prepare/X_train.csv")
y_train = pd.read_csv("prepare/y_train.csv").squeeze()  # 确保是 Series

# 创建 XGBoost 回归器并训练
model = xgb.XGBRegressor(random_state=42, verbosity=0)
model.fit(X_train, y_train)

# 保存模型
os.makedirs("models", exist_ok=True)
model.save_model("models/xgb_model.json")

print("✅ 模型训练完成，已保存为 models/xgb_model.json")
