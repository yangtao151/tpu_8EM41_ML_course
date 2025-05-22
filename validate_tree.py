# validate_tree.py
import pandas as pd
import joblib
import json
import os
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn import tree
import matplotlib.pyplot as plt

# 加载验证集
X_val = pd.read_csv("prepare/X_val.csv")
y_val = pd.read_csv("prepare/y_val.csv").squeeze()

# 加载模型
model = joblib.load("models/tree_model.joblib")

# 预测
y_pred = model.predict(X_val)

# 评估指标
r2 = r2_score(y_val, y_pred)
mae = mean_absolute_error(y_val, y_pred)
mse = mean_squared_error(y_val, y_pred)
rmse = mse ** 0.5

metrics = {
    "R2": round(r2, 4),
    "MAE": round(mae, 2),
    "MSE": round(mse, 2),
    "RMSE": round(rmse, 2)
}

# 保存指标
os.makedirs("models", exist_ok=True)
with open("models/tree_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("✅ 验证指标如下：")
print(metrics)

# 绘制决策树结构图
plt.figure(figsize=(20, 10))
tree.plot_tree(
    model,
    filled=True,
    feature_names=X_val.columns,
    rounded=True,
    max_depth=3  # 限制可视深度以清晰展示
)
plt.tight_layout()
plt.savefig("models/tree_structure.png")
plt.show()
