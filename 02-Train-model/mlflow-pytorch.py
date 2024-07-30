import os
import mlflow
import mlflow.sklearn

import torch

from torch import nn
from torch.utils.data import DataLoader
from torchinfo import summary
from torchmetrics import Accuracy
from torchvision import datasets
from torchvision.transforms import ToTensor
from config import CFG

# Set MLflow tracking URI
os.environ["AWS_ACCESS_KEY_ID"] = CFG.minio_access_key
os.environ["AWS_SECRET_ACCESS_KEY"] = CFG.minio_secret_key

os.environ['MLFLOW_S3_ENDPOINT_URL'] = f'http://{CFG.minio_ip}:{CFG.minio_port}'

device = "cuda" if torch.cuda.is_available() else "cpu"


def get_run_id():
    mlflow.set_tracking_uri(CFG.mlflow_url)
    
    experiment_name = CFG.experiment_name
    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
        print(f"Created new experiment with ID: {experiment_id}")
    else:
        experiment_id = experiment.experiment_id
        print(f"Using existing experiment with ID: {experiment_id}")

    mlflow.set_experiment(experiment_name)
    
    # with ~ as 구문은 end_run()이 없어도 자동 종료    
    # with mlflow.start_run(run_name=experiment_name) as run:
        # run_id = run.info.run_id
        # print(run_id)
        
    run_id = mlflow.start_run(run_name=experiment_name).info.run_id

    experiment_url = f'{CFG.mlflow_url}/#/experiments/{experiment_id}/runs/{run_id}'
    
    return experiment_id, run_id, experiment_url


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )
        
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        
        return logits   


def train(data_loader, model, loss_fn, metrics_fn, optimizer):
    model.train()
    
    for batch, (X, y) in enumerate(data_loader):
        X, y = X.to(device), y.to(device)
        
        pred = model(X)
        loss = loss_fn(pred, y)
        accuracy = metrics_fn(pred, y)
        
        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        if batch % 100 == 0:
            loss, current = loss.item(), batch
            mlflow.log_metric("loss", f"{loss:.3f}", step=(batch // 100))
            mlflow.log_metric("accuracy", f"{accuracy:3f}", step=(batch // 100))
            print(f"loss: {loss:3f} accuracy: {accuracy:3f} [{current} / {len(data_loader)}]")


def main():
    # Download training data from open datasets.
    train_data = datasets.FashionMNIST(
        root="./02-Train-model/data",
        train=True,
        download=True,
        transform=ToTensor(),
    )
    
    train_dataloader = DataLoader(train_data, batch_size=64)
    
    epochs = 3
    loss_fn = nn.CrossEntropyLoss()
    metric_fn = Accuracy(task="multiclass", num_classes=10).to(device)
    model = NeuralNetwork().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    experiment_id, run_id, experiment_url = get_run_id()
    print(experiment_id)
    print(experiment_url)
    
    with mlflow.start_run(run_id=run_id, run_name=CFG.experiment_name, nested=True) as run:
        run_id = run.info.run_id
        
        params = {
            "epochs": epochs,
            "learning_rate": 1e-3,
            "batch_size": 64,
            "loss_function": loss_fn.__class__.__name__,
            "metric_function": metric_fn.__class__.__name__,
            "optimizer": "SGD",
        }

        # Log training parameters.
        mlflow.log_params(params)

        # Log model summary.
        with open("./02-Train-model/model_summary.txt", "w") as f:
            f.write(str(summary(model)))
            
        mlflow.log_artifact("./02-Train-model/model_summary.txt")

        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            train(train_dataloader, model, loss_fn, metric_fn, optimizer)

        # Save the trained model to MLflow.
        mlflow.pytorch.log_model(model, "model")



if __name__ == '__main__':
    main()