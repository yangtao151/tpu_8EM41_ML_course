import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import os
import yaml
import json
from torch.utils.tensorboard import SummaryWriter

# ========== 1. 加载标准化数据 ==========
X_train = pd.read_csv("prepare/X_train_scaled.csv")
y_train = pd.read_csv("prepare/y_train.csv").squeeze()

# ========== 2. 从 params.yaml 读取超参数 ==========
with open("params.yaml") as f:
    params = yaml.safe_load(f)["mlp"]

model = MLPRegressor(
    hidden_layer_sizes=tuple(params["hidden_layer_sizes"]),
    activation=params["activation"],
    solver=params["solver"],
    max_iter=params["max_iter"],
    random_state=params["random_state"],
    early_stopping=True,
    verbose=False
)

# ========== 3. 模型训练 ==========
model.fit(X_train, y_train)

# ========== 4. 创建目录 ==========
os.makedirs("models", exist_ok=True)
os.makedirs("metrics", exist_ok=True)
os.makedirs("runs/mlp", exist_ok=True)

# ========== 5. 保存模型 ==========
joblib.dump(model, "models/mlp_model.pkl")

# ========== 6. 输出指标 ==========
y_pred = model.predict(X_train)
metrics = {
    "R2": float(r2_score(y_train, y_pred)),
    "MAE": float(mean_absolute_error(y_train, y_pred)),
    "MSE": float(mean_squared_error(y_train, y_pred)),
    "RMSE": float(np.sqrt(mean_squared_error(y_train, y_pred)))
}
with open("metrics/mlp_train_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

# ========== 7. 绘制 Loss 曲线 ==========
plt.figure(figsize=(8, 5))
plt.plot(model.loss_curve_)
plt.title("MLP Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.tight_layout()
plt.savefig("models/mlp_loss_curve.png")
plt.close()

# ========== 8. 权重直方图 ==========
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

# ========== 9. 写入 TensorBoard ==========
writer = SummaryWriter(log_dir="runs/mlp")
for i, loss in enumerate(model.loss_curve_):
    writer.add_scalar("Loss/train", loss, i)
writer.add_histogram("MLP Weights", all_weights, 0)
writer.close()

print("✅ train_mlp.py 执行完毕，模型（.pkl）、图像（models/）、指标（JSON）、TensorBoard 日志已全部生成。")
