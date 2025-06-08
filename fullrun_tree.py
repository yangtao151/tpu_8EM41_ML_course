import pandas as pd
import joblib
import os
import json
import numpy as np
from sklearn.tree import plot_tree
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# 创建目录
os.makedirs("metrics", exist_ok=True)
os.makedirs("models", exist_ok=True)

# 加载完整数据
X = pd.read_csv("prepare/X_full.csv")
y = pd.read_csv("prepare/y_full.csv").squeeze()

# 加载已训练模型
model = joblib.load("models/tree_model.pkl")

# 模型预测
y_pred = model.predict(X)

# 计算评估指标
metrics = {
    "R2": float(r2_score(y, y_pred)),
    "MAE": float(mean_absolute_error(y, y_pred)),
    "MSE": float(mean_squared_error(y, y_pred)),
    "RMSE": float(np.sqrt(mean_squared_error(y, y_pred)))
}

# 保存指标
with open("metrics/tree_full_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("✅ 决策树全数据评估指标如下：")
for k, v in metrics.items():
    print(f"{k}: {v}")

# 保存结构图
plt.figure(figsize=(24, 12))
plot_tree(
    model,
    filled=True,
    feature_names=X.columns,
    rounded=True,
    max_depth=3
)
plt.tight_layout()
plt.savefig("models/tree_structure_full.png")
plt.show()
