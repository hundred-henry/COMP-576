# -*- coding: utf-8 -*-
"""Assignment_2_Part_2_RNN_MNIST_vp1.ipynb
Overall structure:

1) Set Pytorch metada
- seed
- tensorflow output
- whether to transfer to gpu (cuda)

2) Import data
- download data
- create data loaders with batchsize, transforms, scaling

3) Define Model architecture, loss and optimizer

4) Define Test and Training loop
    - Train:
        a. get next batch
        b. forward pass through model
        c. calculate loss
        d. backward pass from loss (calculates the gradient for each parameter)
        e. optimizer: performs weight updates

5) Perform Training over multiple epochs:
    Each epoch:
    - call train loop
    - call test loop

# Step 1: Pytorch and Training Metadata
"""
from tkinter.constants import HIDDEN

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
from pathlib import Path
import matplotlib.pyplot as plt

batch_size = 64
test_batch_size = 1000
epochs = 15
lr = 0.005
try_cuda = True
seed = 1000
logging_interval = 50 # how many batches to wait before logging
logging_dir = None

INPUT_SIZE = 28
HIDDEN_SIZE = 64
NUM_LAYERS = 1
BIDIRECTIONAL = True

# 1) setting up the logging

datetime_str = datetime.now().strftime('%b%d_%H-%M-%S')

if logging_dir is None:
    runs_dir = Path("./") / Path(f"runs/mnist")
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

# Setting up data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

# plot one example
print(train_dataset.data.size())     # (60000, 28, 28)
print(train_dataset.targets.size())   # (60000)
plt.imshow(train_dataset.data[0].numpy(), cmap='gray')
plt.title('%i' % train_dataset.targets[0])
plt.show()

"""# Step 3: Creating the Model"""

class Net(nn.Module):
    def __init__(self, input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, bidirectional=False,
                 model_type='rnn'):
        super(Net, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.model_type = model_type
        if model_type == 'rnn' or model_type == 'RNN':
            self.rnn = nn.RNN(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=bidirectional
            )
        elif model_type == 'gru' or model_type == 'GRU':
            self.rnn = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=bidirectional
            )
        elif model_type == 'lstm' or model_type == 'LSTM':
            self.rnn = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=bidirectional
            )
        else:
            raise ValueError("Model should be 'rnn', 'gru', or 'lstm'")
        self.out = nn.Linear(64 * (2 if bidirectional else 1), 10)

    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
        r_out, hidden = self.rnn(x, None)   # None represents zero initial hidden state

        # choose r_out at the last time step
        out = self.out(r_out[:, -1, :])
        return out

model = Net(model_type='lstm')

if cuda:
    model.cuda()

# optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=0.0001)
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001)

"""# Step 4: Train/Test"""

# Defining the test and training loops

def train(epoch):
    model.train()

    criterion = nn.CrossEntropyLoss()
    for batch_idx, (data, target) in enumerate(train_loader):
        if cuda:
            data, target = data.cuda(), target.cuda()

        data = data.view(-1, 28, 28)

        optimizer.zero_grad()
        output = model(data) # forward
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % logging_interval == 0:
            print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} "
                  f"({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}")
            niter = epoch * len(train_loader) + batch_idx
            writer.add_scalar('Loss/train', loss.item(), niter)



def test(epoch):
    model.eval()

    criterion = nn.CrossEntropyLoss()
    test_loss = 0.0
    correct = 0.0
    with torch.no_grad():
        for data, target in test_loader:
            if cuda:
                data, target = data.cuda(), target.cuda()

            data = data.view(-1, 28, 28)

            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n")
    writer.add_scalar('Loss/test', test_loss, epoch)
    writer.add_scalar('Accuracy/test', accuracy, epoch)

# Training loop

for epoch in range(1, epochs + 1):
    train(epoch)
    test(epoch)

    # lr decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * 0.95

writer.close()

# Commented out IPython magic to ensure Python compatibility.
"""
#https://stackoverflow.com/questions/55970686/tensorboard-not-found-as-magic-function-in-jupyter

#seems to be working in firefox when not working in Google Chrome when running in Colab
#https://stackoverflow.com/questions/64218755/getting-error-403-in-google-colab-with-tensorboard-with-firefox


# %load_ext tensorboard
# %tensorboard --logdir [dir]

"""