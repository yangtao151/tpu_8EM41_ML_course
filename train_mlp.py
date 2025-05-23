import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import os

from torch.utils.tensorboard import SummaryWriter

# ========== 1. 读取训练数据 ==========
X_train = pd.read_csv("prepare/X_train.csv")
y_train = pd.read_csv("prepare/y_train.csv")

# ========== 2. 训练模型 ==========
model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42, verbose=False)
model.fit(X_train, y_train.values.ravel())

# ========== 3. 保存模型 ==========
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/mlp_model.joblib")

# ========== 4. 绘制 loss 曲线 ==========
plt.figure(figsize=(8, 5))
plt.plot(model.loss_curve_)
plt.title("MLP Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.tight_layout()
plt.savefig("models/mlp_loss_curve.png")
plt.close()

# ========== 5. 绘制权重直方图 ==========
coefs = model.coefs_
all_weights = np.concatenate([w.flatten() for w in coefs])

plt.figure(figsize=(8, 5))
plt.hist(all_weights, bins=50, color='skyblue', edgecolor='black')
plt.title("MLP Weight Distribution")
plt.xlabel("Weight Value")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.savefig("models/mlp_weight_hist.png")
plt.close()

# ========== 6. TensorBoard 输出 ==========
writer = SummaryWriter(log_dir="runs/mlp")

if hasattr(model, "loss_curve_"):
    for i, loss in enumerate(model.loss_curve_):
        writer.add_scalar("Loss/train", loss, i)

writer.add_histogram("MLP Weights", all_weights, 0)
writer.close()

print("✅ train_mlp.py 执行完毕，模型、图像、TensorBoard 日志均已生成。")
