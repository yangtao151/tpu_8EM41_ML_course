# train_tree.py
import pandas as pd
import joblib
import os
from sklearn.tree import DecisionTreeRegressor

# 加载训练集
X_train = pd.read_csv("prepare/X_train.csv")
y_train = pd.read_csv("prepare/y_train.csv").squeeze()  # 确保是一维 Series

# 创建决策树模型并训练
model = DecisionTreeRegressor(random_state=42)
model.fit(X_train, y_train)

# 保存模型
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/tree_model.joblib")

print("✅ 决策树模型训练完成，已保存为 models/tree_model.joblib")
