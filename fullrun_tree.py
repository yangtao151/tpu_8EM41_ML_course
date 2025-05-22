# fullrun_tree.py
import pandas as pd
import joblib
import os
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# 加载全量数据
X = pd.read_csv("prepare/X_full.csv")
y = pd.read_csv("prepare/y_full.csv").squeeze()

# 模型训练
model = DecisionTreeRegressor(random_state=42)
model.fit(X, y)

# 预测
y_pred = model.predict(X)

# 指标
r2 = r2_score(y, y_pred)
mae = mean_absolute_error(y, y_pred)
mse = mean_squared_error(y, y_pred)
rmse = mse ** 0.5

print("✅ 决策树全数据评估结果：")
print(f"R²   = {r2:.4f}")
print(f"MAE  = {mae:.2f}")
print(f"MSE  = {mse:.2f}")
print(f"RMSE = {rmse:.2f}")

# 保存结构图
os.makedirs("models", exist_ok=True)
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
