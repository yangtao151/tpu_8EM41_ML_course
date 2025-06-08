import pandas as pd
import joblib
import json
import os
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn import tree
import matplotlib.pyplot as plt
import numpy as np

# 加载验证数据
X_val = pd.read_csv("prepare/X_val.csv")
y_val = pd.read_csv("prepare/y_val.csv").squeeze()

# 加载模型（使用 .pkl）
model = joblib.load("models/tree_model.pkl")

# 预测
y_pred = model.predict(X_val)

# 计算指标
metrics = {
    "R2": float(r2_score(y_val, y_pred)),
    "MAE": float(mean_absolute_error(y_val, y_pred)),
    "MSE": float(mean_squared_error(y_val, y_pred)),
    "RMSE": float(np.sqrt(mean_squared_error(y_val, y_pred)))
}

# 保存指标
os.makedirs("metrics", exist_ok=True)
with open("metrics/tree_val_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("✅ 决策树验证指标如下：")
print(metrics)

# 🌳 决策树结构图
plt.figure(figsize=(20, 10))
tree.plot_tree(
    model,
    filled=True,
    feature_names=X_val.columns,
    rounded=True,
    max_depth=3
)
plt.tight_layout()
plt.savefig("models/tree_structure.png")
plt.show()
