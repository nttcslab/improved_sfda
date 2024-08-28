
from functools import partial
from typing import Any, Callable, List, Optional, Type, Union, Tuple

import torch
from torch import Tensor
from torch.autograd import Variable
from torchvision.models.resnet import BasicBlock, Bottleneck, ResNet, ResNet50_Weights, ResNet101_Weights
from torchvision.models._api import WeightsEnum
from torchvision.models._utils import _ovewrite_named_param


class ResNet_AMO_Feat(ResNet):
    def forward_perturb(self, x: Tensor, deltas: List[Tensor]) -> Tensor:
        # See note [TorchScript super()]
        bs = x.size(0)

        x = self.conv1(x)
        x = x + torch.norm(x.view(bs, -1), dim=1)[:, None, None, None] * deltas[0]
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = x + torch.norm(x.view(bs, -1), dim=1)[:, None, None, None] * deltas[1]
        x = self.layer2(x)
        x = x + torch.norm(x.view(bs, -1), dim=1)[:, None, None, None] * deltas[2]
        x = self.layer3(x)
        x = x + torch.norm(x.view(bs, -1), dim=1)[:, None, None, None] * deltas[3]
        x = self.layer4(x)
        x = x + torch.norm(x.view(bs, -1), dim=1)[:, None, None, None] * deltas[4]

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return x

    def forward_init_delta(self, x):
        bs = x.size(0)

        x = self.conv1(x)
        delta_0 = Variable(torch.zeros(x.size(), dtype=torch.float32,device=x.device), requires_grad=True)
        x = x + torch.norm(x.view(bs, -1), dim=1)[:, None, None, None] * delta_0
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        delta_1 = Variable(torch.zeros(x.size(), dtype=torch.float32,device=x.device), requires_grad=True)
        x = x + torch.norm(x.view(bs, -1), dim=1)[:, None, None, None] * delta_1

        x = self.layer2(x)
        delta_2 = Variable(torch.zeros(x.size(), dtype=torch.float32,device=x.device), requires_grad=True)
        x = x + torch.norm(x.view(bs, -1), dim=1)[:, None, None, None] * delta_2

        x = self.layer3(x)
        delta_3 = Variable(torch.zeros(x.size(), dtype=torch.float32,device=x.device), requires_grad=True)
        x = x + torch.norm(x.view(bs, -1), dim=1)[:, None, None, None] * delta_3

        x = self.layer4(x)
        delta_4 = Variable(torch.zeros(x.size(), dtype=torch.float32,device=x.device), requires_grad=True)
        x = x + torch.norm(x.view(bs, -1), dim=1)[:, None, None, None] * delta_4

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return x, [delta_0, delta_1, delta_2, delta_3, delta_4]

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return x

def _resnet_amo(
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    weights: Optional[WeightsEnum],
    progress: bool,
    **kwargs: Any,
) -> ResNet_AMO_Feat:
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = ResNet_AMO_Feat(block, layers, **kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    return model


def resnet50_amo(pretrained: bool = True, progress: bool = True, **kwargs: Any) -> ResNet_AMO_Feat:
    if pretrained:
        weights = ResNet50_Weights.IMAGENET1K_V1
    else:
        weights = None

    weights = ResNet50_Weights.verify(weights)

    net = _resnet_amo(Bottleneck, [3, 4, 6, 3], weights, progress, **kwargs)
    del net.fc
    return net


def resnet101_amo(pretrained: bool = True, progress: bool = True, **kwargs: Any) -> ResNet_AMO_Feat:
    if pretrained:
        weights = ResNet101_Weights.IMAGENET1K_V1
    else:
        weights = None

    weights = ResNet101_Weights.verify(weights)

    net = _resnet_amo(Bottleneck, [3, 4, 23, 3], weights, progress, **kwargs)
    del net.fc
    return net