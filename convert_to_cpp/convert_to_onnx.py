import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models.resnet import resnet18


class SoftMaxModel(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self._model = model

    def forward(self, batch):
        return F.softmax(self._model(batch), dim=1)


def convert_to_onnx():
    model = SoftMaxModel(resnet18(pretrained=True)).cuda()
    dummy_input = torch.randn(16, 3, 244, 244, device='cuda')

    torch.onnx.export(model, dummy_input, "model.onnx", input_names=['images'], output_names=['cls_prob'])


if __name__ == '__main__':
    convert_to_onnx()
