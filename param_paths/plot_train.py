import pickle
import random
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d import Axes3D
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.models.resnet import resnet18
from torchvision.transforms import Compose, Normalize, ToTensor

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
print_freq = 100
base_lr = 0.01

cache_path = Path('cache.pkl')


def get_param_subset(model: nn.Module, indices: List[int]) -> np.ndarray:
    params = torch.cat([x.view(-1) for x in model.parameters()], dim=0)
    if params.is_cuda:
        params = params.cpu()
    return params.detach().numpy()[indices]


def train():
    if cache_path.exists():
        with cache_path.open('rb') as f:
            param_data = pickle.load(f)
    else:
        model = resnet18(num_classes=10)
        model = nn.DataParallel(model.cuda())

        lr = base_lr
        transform = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset = CIFAR10('./data', train=True, download=True, transform=transform)
        loader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=4)
        dataiter = iter(loader)

        criterion = nn.CrossEntropyLoss()
        optimizer = SGD(model.parameters(), lr, momentum=0.9, weight_decay=0.0005)

        num_params = sum([p.numel() for p in model.parameters()])
        indices = list(range(num_params))
        random.shuffle(indices)
        chosen_indices = indices[:100]
        param_data = []

        total_loss = 0
        model.train()
        for idx in range(6000):
            if idx in [4000]:
                lr *= 0.1
                optimizer = SGD(model.parameters(), lr, momentum=0.9, weight_decay=0.0005)
            try:
                inputs, labels = next(dataiter)
            except StopIteration:
                dataiter = iter(loader)
                inputs, labels = next(dataiter)
            inputs = inputs.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss)

            param_data.append(get_param_subset(model, chosen_indices))

            if idx > 0 and idx % print_freq == 0:
                print(f"Iter={idx:4d} Loss={total_loss / print_freq:.3f}")
                total_loss = 0

        param_data = np.array(param_data)

        with cache_path.open('wb') as f:
            pickle.dump(param_data, f)

    swaped = np.swapaxes(param_data, 0, 1)
    displacements = np.sum(np.abs(swaped[1:] - swaped[:-1]), axis=1)
    order = np.argsort(displacements)[::-1]

    fig = plt.figure(figsize=(14, 12))
    ax = plt.subplot(2, 2, 1)
    for idx, param_idx in enumerate(order):
        param_hist = param_data[:, param_idx]
        ax.plot(param_hist)
        if idx > 10:
            break
    ax.set_title("Values of parameters with great movement")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Parameter values")

    ax = plt.subplot(2, 2, 2)
    xs = param_data[:, order[0]]
    ys = param_data[:, order[1]]
    ax.scatter(xs, ys, s=1)
    ax.set_title("Path of two params with largest movements in the sample")
    ax.set_xlabel("Param 1")
    ax.set_ylabel("Param 2")
    ax.text(xs[0], ys[0], "Start")
    ax.text(xs[-1], ys[-1], "End")

    ax = plt.subplot(2, 2, 3, projection=Axes3D.name)
    zs = param_data[:, order[2]]
    ax.scatter(xs, ys, zs, s=1)
    ax.set_title("Path of three params with largest movements in the sample")
    ax.set_xlabel("Param 1")
    ax.set_ylabel("Param 2")
    ax.set_zlabel("Param 3")
    ax.text(xs[0], ys[0], zs[0], "Start")
    ax.text(xs[-1], ys[-1], zs[-1], "End")

    ax = plt.subplot(2, 2, 4)
    ax.axis([0, 1, 0, 1])
    ax.text(0.5, 0.5,
            "As can be seen on the plots, some parameters\n"
            "have quite a lot of displacement during training.\n"
            "\n"
            "The direction of movement for these parameters\n"
            "also changes quite a few times during training,\n"
            "even when using momentum.\n"
            "\n"
            "As a result parameters can pass through some\n"
            "values multiple times during training.",
            ha='center', va='center', wrap=True, fontsize=16)
    ax.axis('off')

    fig.tight_layout()
    fig.savefig("Paths.png")
    plt.show(fig)


if __name__ == '__main__':
    train()
