import argparse
import logging
from pathlib import Path

import torch
import torch.optim as optim
import torchvision
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, Normalize, RandomCrop, RandomHorizontalFlip, ToTensor
from tqdm import tqdm
from yacs.config import CfgNode as CN

from config import load_cfg
from model_builder import create_model


def train(model: nn.Module, train_cfg: CN, train_set: Dataset):
    train_loader = DataLoader(train_set, batch_size=train_cfg.BATCH_SIZE, shuffle=train_cfg.SHUFFLE, num_workers=4)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=train_cfg.LR, momentum=train_cfg.MOMENTUM,
                          weight_decay=train_cfg.WEIGHT_DECAY)

    data_loader = iter(train_loader)

    logging.info('Starting Training')
    smooth_loss = 4.6  # Loss for random prediction of 100 classes
    for idx in tqdm(range(train_cfg.NUM_ITERS), desc="Training"):
        try:
            batch, labels = next(data_loader)
        except StopIteration:
            data_loader = iter(train_loader)
            batch, labels = next(data_loader)

        batch = batch.cuda()
        labels = labels.cuda()

        outputs = model(batch)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        smooth_loss = smooth_loss * 0.99 + float(loss) * 0.01

        if (idx + 1) % 100 == 0:
            logging.debug(f"{idx + 1:4d}: {smooth_loss:.3f} ({loss:.3f})")

    logging.info('Finished Training')


def val(model: nn.Module, val_cfg: CN, val_set: Dataset):
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=val_cfg.BATCH_SIZE, num_workers=4)

    correct = 0
    total = 0

    class_correct = [0. for _ in range(100)]
    class_total = [0. for _ in range(100)]
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validating"):
            images = images.cuda()
            labels = labels.cuda()
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            c = (predicted == labels).squeeze()
            for idx in range(labels.size(0)):
                label = labels[idx]
                class_correct[label] += c[idx].item()
                class_total[label] += 1

    logging.info(f"Accuracy of the network on the 10000 test images: {correct / total:.1%}")


def main(cfg: CN):
    logging.info(f"Using {cfg.NAME} configuration")

    cache_path = Path(cfg.OUTPUT_DIR) / "final.model"
    model = create_model(cfg.MODEL)
    model = nn.DataParallel(model.cuda())

    transform_list = [RandomCrop(32, padding=4), RandomHorizontalFlip(), ToTensor(),
                      Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
    if not cache_path.exists():
        train_set = torchvision.datasets.CIFAR100(root='./data', train=True, download=True,
                                                  transform=Compose(transform_list))
        train(model, cfg.TRAIN, train_set)

        torch.save(model.module.state_dict(), cache_path)

    model.module.load_state_dict(torch.load(cache_path))

    val_set = torchvision.datasets.CIFAR100(root='./data', train=False, download=True,
                                            transform=Compose(transform_list[-2:]))
    val(model, cfg.VAL, val_set)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('cfg', type=Path)
    args = parser.parse_args()

    # Ensure reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    _cfg = load_cfg(args.cfg)

    logging
    file_handler = logging.FileHandler(Path(_cfg.OUTPUT_DIR) / 'train_val.log')
    file_handler.setLevel(logging.DEBUG)

    stdout_handler = logging.StreamHandler()
    stdout_handler.setLevel(logging.INFO)

    logging.basicConfig(handlers=[file_handler, stdout_handler], level=logging.DEBUG,
                        format="{asctime} [{levelname:^9s}] {message}", style="{",
                        datefmt="%Y-%m-%d %H:%M:%S")

    main(_cfg)
