import random 
import numpy as np
from functools import partial
import random 
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
from ray import tune
from ray.tune.search.basic_variant import BasicVariantGenerator

class FashionCNN(nn.Module):
    def __init__(self):
        super(FashionCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
def load_data(data_dir="./data"):
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))])

    trainset = torchvision.datasets.FashionMNIST(data_dir,
        download=True,
        train=True,
        transform=transform)
    testset = torchvision.datasets.FashionMNIST(data_dir,
        download=True,
        train=False,
        transform=transform)
    return trainset, testset

def train_fashion_mnist(config, trainloader, valloader):
    net = FashionCNN() 

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=config["learning_rate"])

    for epoch in range(config["epochs"]):
        running_loss = 0.0
        epoch_steps = 0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            epoch_steps += 1

            if i % 2000 == 1999:
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / epoch_steps))
                running_loss = 0.0

        val_loss = 0.0
        val_steps = 0
        total = 0
        correct = 0
        with torch.no_grad():
            for data in valloader:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_steps += 1

        tune.report(
            mean_accuracy=correct / total,
            mean_val_loss=val_loss / val_steps
        )   
if __name__ == "__main__":
    torch.manual_seed(40)
    random.seed(40)
    np.random.seed(40)

    trainset, testset = load_data()
    test_abs = int(len(trainset) * 0.8)
    train_subset, val_subset = random_split(trainset, [test_abs, len(trainset) - test_abs])
    trainloader = torch.utils.data.DataLoader(
        train_subset, batch_size=64, shuffle=True, num_workers=2
    )
    valloader = torch.utils.data.DataLoader(
        val_subset, batch_size=64, shuffle=True, num_workers=2
    )

    config = {
        "epochs": tune.choice([5, 10, 15]),
        "learning_rate": tune.loguniform(1e-4, 1e-2), 
        "batch_size": tune.choice([16, 32, 64, 128])
    }
    max_num_epochs = 15
    num_samples = 10

    basic_result = tune.run(
        partial(train_fashion_mnist, trainloader=trainloader, valloader=valloader),
        resources_per_trial={"cpu": 5, "gpu": 0.5},
        config=config,
        num_samples=num_samples,
        storage_path='./tune_runs/',
        search_alg=BasicVariantGenerator(random_state=40))

    best_trial = basic_result.get_best_trial("mean_val_loss", mode="min")
    best_config = best_trial.config
    best_metrics = best_trial.metric_analysis

    print("Best trial config:", best_config)
    print("Best trial metrics:", best_metrics)