from typing import List, Dict
import torch.nn as nn
import torch
from torch.nn import Parameter
from torch.nn.utils import weight_norm


class WeightNorm(nn.Module):
    append_g = '_g'
    append_v = '_v'

    def __init__(self, module, weights):
        super(WeightNorm, self).__init__()
        self.module = module
        self.weights = weights
        self._reset()

    def _reset(self):
        for name_w in self.weights:
            w = getattr(self.module, name_w)

            # construct g,v such that w = g/||v|| * v
            g = torch.norm(w)
            v = w/g.expand_as(w)
            g = Parameter(g.data)
            v = Parameter(v.data)
            name_g = name_w + self.append_g
            name_v = name_w + self.append_v

            # remove w from parameter list
            del self.module._parameters[name_w]

            # add g and v as new parameters
            self.module.register_parameter(name_g, g)
            self.module.register_parameter(name_v, v)

    def _setweights(self):
        for name_w in self.weights:
            name_g = name_w + self.append_g
            name_v = name_w + self.append_v
            g = getattr(self.module, name_g)
            v = getattr(self.module, name_v)
            w = v*(g/torch.norm(v)).expand_as(v)
            setattr(self.module, name_w, w)

    def forward(self, *args):
        self._setweights()
        return self.module.forward(*args)


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)


class BaseBottleneckVisda(nn.Module):
    def __init__(self, in_features, out_features, apply_init=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.bn = nn.BatchNorm1d(self.out_features, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        self.bottleneck = nn.Linear(self.in_features, self.out_features)
        self.bottleneck.apply(init_weights)

    def forward(self, x):
        x = self.bottleneck(x)
        x = self.bn(x)
        return x

    @property
    def _features_dim(self):
        return self.out_features


class BaseBottleneck(nn.Module):
    def __init__(self, in_features, out_features, apply_init=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.model = nn.Sequential(
            nn.Linear(self.in_features, self.out_features),
            nn.BatchNorm1d(self.out_features),
        )

        if apply_init:
            for m in self.model:
                classname = m.__class__.__name__
                if classname.find('Linear') != -1:
                    nn.init.xavier_normal_(m.weight)
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.model(x)

    @property
    def _features_dim(self):
        return self.out_features



class BaseBottleneckReLU(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.model = nn.Sequential(
            nn.Linear(self.in_features, self.out_features),
            nn.BatchNorm1d(self.out_features),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.model(x)

    @property
    def _features_dim(self):
        return self.out_features


class BaseHeadVisda(nn.Module):
    def __init__(self, in_features, num_classes, apply_init=False):
        super().__init__()
        self.fc = weight_norm(nn.Linear(in_features, num_classes), name="weight")
        self.fc.apply(init_weights)

    def forward(self, x):
        return self.fc(x)


class BaseHead(nn.Module):
    def __init__(self, in_features, num_classes, apply_init=False):
        super().__init__()
        fc = nn.Linear(in_features, num_classes)
        if apply_init:
            nn.init.xavier_normal_(fc.weight)
            nn.init.zeros_(fc.bias)
        self.model = WeightNorm(fc, ['weight', 'bias'])

    def forward(self, x):
        return self.model(x)



class Projection(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.BatchNorm1d(in_features),
            nn.ReLU(),
            nn.Linear(in_features, out_features)
        )

    def forward(self, x):
        return self.model(x)