# -*- coding: utf-8 -*-
"""Assignment_2_Part_1_Cifar10_vp1.ipynb

Purpose: Implement image classsification nn the cifar10
dataset using a pytorch implementation of a CNN architecture (LeNet5)

Pseudocode:
1) Set Pytorch metada
- seed
- tensorboard output (logging)
- whether to transfer to gpu (cuda)

2) Import the data
- download the data
- create the pytorch datasets
    scaling
- create pytorch dataloaders
    transforms
    batch size

3) Define the model architecture, loss and optimizer

4) Define Test and Training loop
    - Train:
        a. get next batch
        b. forward pass through model
        c. calculate loss
        d. backward pass from loss (calculates the gradient for each parameter)
        e. optimizer: performs weight updates
        f. Calculate accuracy, other stats
    - Test:
        a. Calculate loss, accuracy, other stats

5) Perform Training over multiple epochs:
    Each epoch:
    - call train loop
    - call test loop




"""

# Step 1: Pytorch and Training Metadata

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
from pathlib import Path

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# hyperparameters
batch_size = 128
epochs = 15
lr = 0.002
try_cuda = True
seed = 1000

# Architecture
num_classes = 10

# otherum
logging_interval = 10  # how many batches to wait before logging
logging_dir = None
grayscale = True

# 1) setting up the logging

datetime_str = datetime.now().strftime('%b%d_%H-%M-%S')

if logging_dir is None:
    runs_dir = Path("./") / Path(f"runs/cifar10/")
    runs_dir.mkdir(exist_ok = True)

    logging_dir = runs_dir / Path(f"{datetime_str}")

    logging_dir.mkdir(exist_ok = True)
    logging_dir = str(logging_dir.absolute())

writer = SummaryWriter(log_dir=logging_dir)

#deciding whether to send to the cpu or not if available
if torch.cuda.is_available() and try_cuda:
    cuda = True
    torch.cuda.manual_seed(seed)
else:
    cuda = False
    torch.manual_seed(seed)

"""# Step 2: Data Setup"""


class CIFAR10Dataset(Dataset):
    def __init__(self, dataset_path: str, subset: str = 'Train', transform=None):
        self.dataset_path = dataset_path
        self.subset = subset
        self.transform = transform
        self.images = []
        self.labels = []
        self._load_images()

    def _load_images(self):
        subset_path = os.path.join(self.dataset_path, self.subset)

        # Loop through each class folder (0-9)
        for class_folder in sorted(os.listdir(subset_path)):
            class_path = os.path.join(subset_path, class_folder)

            if os.path.isdir(class_path):
                class_label = int(class_folder)

                for img_file in os.listdir(class_path):
                    img_path = os.path.join(class_path, img_file)
                    try:
                        img = Image.open(img_path).convert('RGB')  # Convert to RGB
                        self.images.append(img)
                        self.labels.append(class_label)
                    except Exception as e:
                        print(f"Error loading image {img_path}: {e}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        if index < 0 or index >= len(self.images):
            raise IndexError("Index out of range.")

        image = self.images[index]
        label = self.labels[index]

        if self.transform:
            image = self.transform(image)

        return image, label

if grayscale:
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.Normalize((0.4822,), (0.2379,))
    ])
else:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])

# Instantiate datasets with transformations
train_dataset = CIFAR10Dataset('./CIFAR10', subset='Train', transform=transform)
test_dataset = CIFAR10Dataset('./CIFAR10', subset='Test', transform=transform)


train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

def check_data_loader_dim(loader):
    # Checking the dataset
    for images, labels in loader:
        print('Image batch dimensions:', images.shape)
        print('Image label dimensions:', labels.shape)
        break

check_data_loader_dim(train_loader)
check_data_loader_dim(test_loader)

"""# 3) Creating the Model"""

layer_1_n_filters = 32
layer_2_n_filters = 64
fc_1_n_nodes = 1024
kernel_size = 5
# padding = kernel_size // 2
padding = "same"
verbose = False

# calculating the side length of the final activation maps
final_length = 7

if verbose:
    print(f"final_length = {final_length}")

class LeNet5(nn.Module):

    def __init__(self, num_classes, grayscale=False):
        super(LeNet5, self).__init__()

        self.grayscale = grayscale
        self.num_classes = num_classes

        if self.grayscale:
            in_channels = 1
        else:
            in_channels = 3

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, layer_1_n_filters, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(layer_1_n_filters, layer_2_n_filters, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(final_length*final_length*layer_2_n_filters*in_channels, fc_1_n_nodes),
            nn.Tanh(),
            nn.Linear(fc_1_n_nodes, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        logits = self.classifier(x)
        probs = F.softmax(logits, dim=1)
        return logits, probs

model = LeNet5(num_classes, grayscale)

if cuda:
    model.cuda()

# optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=0.0001)
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001)

"""# Step 4: Train/Test Loop"""

# Defining the test and training loops

def train(epoch):
    model.train()

    warmup_steps = 500
    base_lr = lr * 0.1

    criterion = nn.CrossEntropyLoss()
    for batch_idx, (data, target) in enumerate(train_loader):
        # warming up
        if epoch == 1 and batch_idx < warmup_steps:
            for param_group in optimizer.param_groups:
                param_group['lr'] = base_lr + (lr - base_lr) * (batch_idx / warmup_steps)


        if cuda:
            data, target = data.cuda(), target.cuda()

        optimizer.zero_grad()
        logits, probs = model(data) # forward
        
        loss = criterion(logits, target)
        loss.backward()
        optimizer.step()

        if batch_idx % logging_interval == 0:
            print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} "
                  f"({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}")
            niter = epoch * len(train_loader) + batch_idx
            writer.add_scalar('Loss/train', loss.item(), niter)

    # [insert-code: Log model parameters to TensorBoard at every epoch]
    for name, param in model.named_parameters():
        layer, attr = os.path.splitext(name)
        attr = attr[1:]
        writer.add_histogram(f'{layer}/{attr}', param.clone().cpu().data.numpy(), epoch)

def test(epoch):
    model.eval()
    test_loss = 0.0
    correct = 0
    criterion = nn.CrossEntropyLoss(reduction='sum')
    with torch.no_grad():
        for data, target in test_loader:
            if cuda:
                data, target = data.cuda(), target.cuda()

            logits, probs  = model(data)

            test_loss += criterion(logits, target).item()
            pred = probs.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print("\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n"
          .format(test_loss, correct, len(test_loader.dataset), accuracy))
    # [insert-code: Log test loss and accuracy to TensorBoard at every epoch]
    writer.add_scalar('Loss/test', test_loss, epoch)
    writer.add_scalar('Accuracy/test', correct, epoch)

for epoch in range(1, epochs + 1):
    train(epoch)
    test(epoch)

    # lr decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * 0.9

writer.close()

def visualize_filters():
    # Get the weights of the first convolutional layer
    # Assuming 'model' is your LeNet5 model
    first_conv_layer = model.features[0]
    weights = first_conv_layer.weight.data.cpu().numpy()

    # Plot the weights as grayscale images
    fig, axes = plt.subplots(4, 8, figsize=(10, 5))  # Adjust the grid size as needed
    fig.suptitle('First Convolutional Layer Filters')

    for i, ax in enumerate(axes.flat):
        if i < weights.shape[0]:  # Only plot existing filters
            # Each filter is of shape (in_channels, height, width), so for grayscale in_channels=1
            img = weights[i, 0, :, :] if model.grayscale else weights[i].mean(axis=0)
            ax.imshow(img, cmap='gray')
            ax.axis('off')
        else:
            ax.axis('off')

    plt.show()

visualize_filters()

# Commented out IPython magic to ensure Python compatibility.
"""
#https://stackoverflow.com/questions/55970686/tensorboard-not-found-as-magic-function-in-jupyter

#seems to be working in firefox when not working in Google Chrome when running in Colab
#https://stackoverflow.com/questions/64218755/getting-error-403-in-google-colab-with-tensorboard-with-firefox


# %load_ext tensorboard
# %tensorboard --logdir [dir]

"""

