import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

# --- загрузка уже векторизованных данных ---
train_df = pd.read_csv(DATA_DIR / "train_split.csv")
test_df = pd.read_csv(DATA_DIR / "test_split.csv")

X_tr = train_df.drop(columns=["label"]).values.astype(np.float32)
y_tr = train_df["label"].values.astype(np.int64)
X_te = test_df.drop(columns=["label"]).values.astype(np.float32)
y_te = test_df["label"].values.astype(np.int64)

n_features = X_tr.shape[1]
n_classes = int(max(y_tr.max(), y_te.max())) + 1

X_tr_t = torch.tensor(X_tr)
y_tr_t = torch.tensor(y_tr)
X_te_t = torch.tensor(X_te)
y_te_t = torch.tensor(y_te)

print(f"train: {X_tr.shape}, test: {X_te.shape}, classes: {n_classes}")


class Net(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, n_out),
        )

    def forward(self, x):
        return self.net(x)


def evaluate(model):
    model.eval()
    with torch.no_grad():
        return (model(X_te_t).argmax(1) == y_te_t).float().mean().item()


# --- обучение с логированием в MLflow ---
def train_once(lr, epochs, hidden=256):
    with mlflow.start_run():
        mlflow.log_param("lr", lr)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("hidden", hidden)
        mlflow.set_tag("task", "doc-classifier")

        model = Net(n_features, n_classes)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()
        losses = []

        for ep in range(epochs):
            model.train()
            opt.zero_grad()
            loss = loss_fn(model(X_tr_t), y_tr_t)
            loss.backward()
            opt.step()
            losses.append(loss.item())
            mlflow.log_metric("loss", loss.item(), step=ep)

        acc = evaluate(model)
        mlflow.log_metric("test_accuracy", acc)

        # артефакт-график
        fig, ax = plt.subplots()
        ax.plot(range(epochs), losses, marker="o")
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        png = f"loss_lr{lr}_ep{epochs}.png"
        fig.savefig(png)
        plt.close(fig)
        mlflow.log_artifact(png)
        os.remove(png)

        # артефакт-конфиг
        mlflow.log_dict({
            "lr": lr, "epochs": epochs, "hidden": hidden,
            "n_features": int(n_features),
            "n_classes": n_classes,
        }, "config.json")

        mlflow.pytorch.log_model(
            model,
            name="model",
            registered_model_name="doc-classifier",
            serialization_format="pickle",
            input_example=X_tr_t[:2],
        )
        return acc


# --- подбор гиперпараметров ---
if __name__ == "__main__":
    mlflow.set_experiment("doc-classifier")

    for lr in [0.0001, 0.001, 0.01]:
        for epochs in [5, 10, 20]:
            acc = train_once(lr, epochs)
            print(f"lr={lr} epochs={epochs} acc={acc:.3f}")

    # --- регистрация лучшей модели и перевод в Staging ---
    client = mlflow.MlflowClient()
    exp = mlflow.get_experiment_by_name("doc-classifier")
    best = client.search_runs(
        [exp.experiment_id],
        order_by=["metrics.test_accuracy DESC"],
        max_results=1,
    )[0]

    versions = client.search_model_versions(f"run_id='{best.info.run_id}'")
    ver = int(versions[0].version)
    try:
        client.transition_model_version_stage("doc-classifier", ver, "Staging")
    except Exception:
        # MLflow 3.x: вместо стадий — алиасы
        client.set_registered_model_alias("doc-classifier", "staging", ver)

    print("BEST run:", best.info.run_id,
          "acc:", best.data.metrics["test_accuracy"],
          "model version:", ver)