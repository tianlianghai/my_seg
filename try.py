import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST, CIFAR10
from torchvision.transforms import ToTensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.parallel import DataParallel as DP
from tqdm import tqdm
from torch import distributed  as dist
from torchvision import transforms, models


import time

# Define your model architecture
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(784, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc(x)
        return x

# Function to train the model
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    start_time = time.time()
  
    for images, labels in tqdm(dataloader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    end_time = time.time()
    train_time = end_time - start_time
    return running_loss, train_time

# Define the main function
def main():
    # Define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the model
    model =models.resnet50(weights="DEFAULT")

    # Load the MNIST dataset
    train_dataset = CIFAR10(root='data/', train=True,
                                  transform=transforms.Compose([
                                      transforms.Resize(32),
                                      transforms.ToTensor()
                                  ]),
                                     download=True)
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # Train using one GPU
#     print("Training with one GPU...")
#     model_single_gpu = model.to(device)
#     loss_single_gpu, time_single_gpu = train(model_single_gpu, train_dataloader, criterion, optimizer, device)
#     print("Loss with one GPU:", loss_single_gpu)
#     print("Training time with one GPU:", time_single_gpu)

    # Train using two GPUs
    if torch.cuda.device_count() > 1:
        dist.init_process_group('nccl')
        rank = dist.get_rank()
        device_id = rank % torch.cuda.device_count()
        print("Training with two GPUs...")
        model_multi_gpu = DDP(model, device_ids=[device_id])
        loss_multi_gpu, time_multi_gpu = train(model_multi_gpu, train_dataloader, criterion, optimizer, device)
        print("Loss with two GPUs:", loss_multi_gpu)
        print("Training time with two GPUs:", time_multi_gpu)
    else:
        print("Two GPUs not available.")

# Execute the main function if this script is run directly
if __name__ == "__main__":
    main()