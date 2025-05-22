# train_mlp.py
import pandas as pd
import os
import joblib
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt

# 加载训练数据
X_train = pd.read_csv("prepare/X_train.csv")
y_train = pd.read_csv("prepare/y_train.csv").squeeze()

# 创建 MLP 模型
model = MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu', solver='adam',
                     max_iter=1000, random_state=42, verbose=False)

# 模型训练
model.fit(X_train, y_train)

# 保存模型
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/mlp_model.joblib")
print("✅ MLP 模型训练完成，已保存为 models/mlp_model.joblib")

# 可视化损失函数收敛曲线
plt.figure(figsize=(8, 5))
plt.plot(model.loss_curve_)
plt.title("MLP Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.tight_layout()
plt.savefig("models/mlp_loss_curve.png")
plt.show()
