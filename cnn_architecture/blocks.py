import torch.nn.functional as F
from torch import Tensor, nn


# Complexity of Conv2d, without feature_map_size: kernel_size^2*in_channels*out_channels

# All blocks should have equal or lower computational complexity than simple block
# Count multiplications only as an approximation
# Total_complexity: 9*in*hidden + 3*9*hidden^2
class SimpleBlock(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self._num_hidden = in_channels * 2
        self._conv1 = nn.Conv2d(in_channels, self._num_hidden, 3, stride=2, padding=1)
        self._conv2 = nn.Conv2d(self._num_hidden, self._num_hidden, 3, padding=1)
        self._conv3 = nn.Conv2d(self._num_hidden, self._num_hidden, 3, padding=1)

    def forward(self, batch: Tensor) -> Tensor:
        batch = F.relu(self._conv1(batch))
        batch = F.relu(self._conv2(batch))
        batch = F.relu(self._conv3(batch))
        return batch

    @property
    def num_hidden(self):
        return self._num_hidden


# Total complexity: 9*in*hidden + 3*9*(hidden*hidden/2+hidden/2*hidden) + 3*relu
class AddBlock(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self._num_hidden = in_channels * 2
        self._stride = nn.Conv2d(in_channels, self._num_hidden, 3, stride=2, padding=1)

        self._conv1 = nn.Conv2d(self._num_hidden, self._num_hidden // 2, 3, padding=1)
        self._adj1 = nn.Conv2d(self._num_hidden // 2, self._num_hidden, 3, padding=1)

        self._conv2 = nn.Conv2d(self._num_hidden, self._num_hidden // 2, 3, padding=1)
        self._adj2 = nn.Conv2d(self._num_hidden // 2, self._num_hidden, 3, padding=1)

    def forward(self, batch: Tensor) -> Tensor:
        batch = F.relu(self._stride(batch))
        batch = batch + self._adj1(F.relu(self._conv1(batch)))
        batch = batch + self._adj2(F.relu(self._conv2(batch)))
        return batch

    @property
    def num_hidden(self):
        return self._num_hidden


# BN Complexity: 2
class BNPreBlock(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self._num_hidden = in_channels * 2
        self._bn1 = nn.BatchNorm2d(in_channels)
        self._conv1 = nn.Conv2d(in_channels, self._num_hidden, 3, stride=2, padding=1)
        self._bn2 = nn.BatchNorm2d(self._num_hidden)
        self._conv2 = nn.Conv2d(self._num_hidden, self._num_hidden - 1, 3, padding=1)
        self._bn3 = nn.BatchNorm2d(self._num_hidden - 1)
        self._conv3 = nn.Conv2d(self._num_hidden - 1, self._num_hidden, 3, padding=1)

    def forward(self, batch: Tensor) -> Tensor:
        batch = F.relu(self._conv1(self._bn1(batch)))
        batch = F.relu(self._conv2(self._bn2(batch)))
        batch = F.relu(self._conv3(self._bn3(batch)))
        return batch

    @property
    def num_hidden(self):
        return self._num_hidden


class BNBetweenBlock(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self._num_hidden = in_channels * 2
        self._conv1 = nn.Conv2d(in_channels, self._num_hidden, 3, stride=2, padding=1)
        self._bn1 = nn.BatchNorm2d(self._num_hidden)
        self._conv2 = nn.Conv2d(self._num_hidden, self._num_hidden - 1, 3, padding=1)
        self._bn2 = nn.BatchNorm2d(self._num_hidden - 1)
        self._conv3 = nn.Conv2d(self._num_hidden - 1, self._num_hidden, 3, padding=1)
        self._bn3 = nn.BatchNorm2d(self._num_hidden)

    def forward(self, batch: Tensor) -> Tensor:
        batch = F.relu(self._bn1(self._conv1(batch)))
        batch = F.relu(self._bn2(self._conv2(batch)))
        batch = F.relu(self._bn3(self._conv3(batch)))
        return batch

    @property
    def num_hidden(self):
        return self._num_hidden


class BNPostBlock(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self._num_hidden = in_channels * 2
        self._conv1 = nn.Conv2d(in_channels, self._num_hidden, 3, stride=2, padding=1)
        self._bn1 = nn.BatchNorm2d(self._num_hidden)
        self._conv2 = nn.Conv2d(self._num_hidden, self._num_hidden - 1, 3, padding=1)
        self._bn2 = nn.BatchNorm2d(self._num_hidden - 1)
        self._conv3 = nn.Conv2d(self._num_hidden - 1, self._num_hidden, 3, padding=1)
        self._bn3 = nn.BatchNorm2d(self._num_hidden)

    def forward(self, batch: Tensor) -> Tensor:
        batch = self._bn1(F.relu(self._conv1(batch)))
        batch = self._bn2(F.relu(self._conv2(batch)))
        batch = self._bn3(F.relu(self._conv3(batch)))
        return batch

    @property
    def num_hidden(self):
        return self._num_hidden
