# from https://github.com/moskomule/dda/blob/master/faster_autoaugment/policy.py

from __future__ import annotations

import random
from copy import deepcopy
from typing import Optional

import torch
from torch import nn, Tensor
from torch.distributions import Categorical

from src.models.components.faster_autoaugment.dda.operations import *

def init_parameters(op: nn.Module):
    for p in op.parameters():
        nn.init.uniform_(p, 0, 1)

class SubPolicyStage(nn.Module):
    def __init__(self,
                 operation_dict: dict,
                 temperature: float,
                 init_operation = None,
                 init_op_probability = 0.9,
                 init_prob_and_mag = None,
                 ):
        super(SubPolicyStage, self).__init__()
        if init_operation is not None:
            init_op, init_prob, init_mag = init_operation # 初期値オペレーション
            init_op_id = -1
            operations = list()
            for i, (op_name, op) in enumerate(operation_dict.items()):
                if op_name == init_op:
                    if init_mag is not None: # magnitudeが存在するオペレーション
                        operations.append(op(initial_probability = init_prob, initial_magnitude = init_mag / 10)) # 10はAutoAugmentのbins
                    else: # magnitudeが存在しないオペレーション(AutoContrastなど)
                        operations.append(op(initial_probability = init_prob))
                    init_op_id = i
                else:
                    operation = op()
                    operations.append(operation) # その他は初期値で設定する
            self.operations = nn.ModuleList(operations)
            _weights = torch.zeros(len(self.operations))
            _weights[init_op_id] = temperature * torch.log(torch.Tensor([init_op_probability * (len(operation_dict) - 1) / (1 - init_op_probability)])) # weightsでサンプルしたときに初期値がpとなるように設定
            self._weights = nn.Parameter(_weights) # 初期値オペレーションに重みをつける。
        elif init_prob_and_mag is not None:
            init_prob, init_mag = init_prob_and_mag
            self.operations = nn.ModuleList([op(initial_probability = init_prob, initial_magnitude = init_mag) for op in operation_dict.values()])
            self._weights = nn.Parameter(torch.ones(len(self.operations)))
        else:
            self.operations = nn.ModuleList([op() for op in operation_dict.values()])
            self._weights = nn.Parameter(torch.ones(len(self.operations)))
            init_parameters(self) # 重みを乱数初期化
        self.temperature = temperature

    def forward(self,
                input: Tensor
                ) -> Tensor:
        if self.training:
            return (torch.stack([op(input) for op in self.operations]) * self.weights.view(-1, 1, 1, 1, 1)).sum(0)
        else:
            return self.operations[Categorical(self.weights).sample()](input)

    @property
    def weights(self
                ):
        return self._weights.div(self.temperature).softmax(0)


class SubPolicy(nn.Module):
    def __init__(self,
                 operation_dict: dict,
                 temperature: float,
                 operation_count: int,
                 init_operations = None,
                 init_op_probability = 0.9,
                 init_prob_and_mag = None,
                 ):
        super(SubPolicy, self).__init__()
        if init_operations is not None:
            self.stages = nn.ModuleList([SubPolicyStage(operation_dict, temperature, init_operation=init_op, init_op_probability=init_op_probability) for init_op in init_operations])
        elif init_prob_and_mag is not None:
            self.stages = nn.ModuleList([SubPolicyStage(operation_dict, temperature, init_prob_and_mag=init_prob_and_mag) for _ in range(operation_count)])
        else:
            self.stages = nn.ModuleList([SubPolicyStage(operation_dict, temperature) for _ in range(operation_count)])

    def forward(self,
                input: Tensor
                ) -> Tensor:
        for stage in self.stages:
            input = stage(input)
        return input


class Policy(nn.Module):
    def __init__(self,
                 operation_dict: dict,
                 num_sub_policies: int,
                 temperature: float = 0.05,
                 operation_count: int = 2,
                 num_chunks: int = 4,
                 init_operations = None,
                 init_op_probability = 0.9,
                 init_prob_and_mag = None):

        super(Policy, self).__init__()
        if init_operations is not None:
            self.sub_policies = nn.ModuleList([SubPolicy(operation_dict=operation_dict, temperature=temperature, operation_count=operation_count, init_operations=op, init_op_probability=init_op_probability) for op in init_operations])
        elif init_prob_and_mag is not None:
            self.sub_policies = nn.ModuleList([SubPolicy(operation_dict=operation_dict, temperature=temperature, operation_count=operation_count, init_prob_and_mag=init_prob_and_mag) for _ in range(num_sub_policies)])
        else:
            self.sub_policies = nn.ModuleList([SubPolicy(operation_dict=operation_dict, temperature=temperature, operation_count=operation_count) for _ in range(num_sub_policies)])
        self.num_sub_policies = num_sub_policies
        self.temperature = temperature
        self.operation_count = operation_count
        self.num_chunks = num_chunks

    def forward(self,
                input: Tensor # not normalized data should be input
                ) -> Tensor:

        if self.num_chunks > 1:
            out = [self._forward(inp) for inp in input.chunk(self.num_chunks)]
            x = torch.cat(out, dim=0)
        else:
            x = self._forward(input)

        return x

    def _forward(self,
                 input: Tensor
                 ) -> Tensor:
        index = random.randrange(self.num_sub_policies)
        return self.sub_policies[index](input)

    @staticmethod
    def dda_operations():

        return {
            "ShearX": ShearX,
            "ShearY": ShearY,
            "TranslateX": TranslateX,
            "TranslateY": TranslateY,
            "Rotate": Rotate,
            "Brightness": Brightness,
            "Color": Saturate,
            "Contrast": Contrast,
            "Sharpness": Sharpness,
            "Posterize": Posterize,
            "Solarize": Solarize,
            "AutoContrast": AutoContrast,
            "Equalize": Equalize,
            "Invert": Invert,
        }
    
    @staticmethod
    def dda_operations_wo_negative():
        return {
            "ShearX": ShearX,
            "ShearY": ShearY,
            "TranslateX": TranslateX,
            "TranslateY": TranslateY,
            "Brightness": Brightness,
            "Color": Saturate,
            "Contrast": Contrast,
            "Sharpness": Sharpness,
            "Posterize": Posterize,
            # "Solarize": Solarize,
            "AutoContrast": AutoContrast,
            # "Equalize": Equalize,
        }

    @staticmethod
    def faster_auto_augment_policy(num_sub_policies: int,
                                   temperature: float,
                                   operation_count: int,
                                   num_chunks: int,
                                   ) -> Policy:

        return Policy(Policy.dda_operations(), num_sub_policies, temperature, operation_count, num_chunks)
    
    @staticmethod
    def init_w_auto_augment_policy(init_op_probability=0.9, num_chunks=4) -> Policy:
        init_operations = [(('Posterize', 0.4, 8), ('Rotate', 0.6, 9)),
                           (('Solarize', 0.6, 5), ('AutoContrast', 0.6, None)),
                           (('Equalize', 0.8, None), ('Equalize', 0.6, None)),
                           (('Posterize', 0.6, 7), ('Posterize', 0.6, 6)),
                           (('Equalize', 0.4, None), ('Solarize', 0.2, 4)),
                           (('Equalize', 0.4, None), ('Rotate', 0.8, 8)),
                           (('Solarize', 0.6, 3), ('Equalize', 0.6, None)),
                           (('Posterize', 0.8, 5), ('Equalize', 1.0, None)),
                           (('Rotate', 0.2, 3), ('Solarize', 0.6, 8)),
                           (('Equalize', 0.6, None), ('Posterize', 0.4, 6)),
                           (('Rotate', 0.8, 8), ('Color', 0.4, 0)),
                           (('Rotate', 0.4, 9), ('Equalize', 0.6, None)),
                           (('Equalize', 0.0, None), ('Equalize', 0.8, None)),
                           (('Invert', 0.6, None), ('Equalize', 1.0, None)),
                           (('Color', 0.6, 4), ('Contrast', 1.0, 8)),
                           (('Rotate', 0.8, 8), ('Color', 1.0, 2)),
                           (('Color', 0.8, 8), ('Solarize', 0.8, 7)),
                           (('Sharpness', 0.4, 7), ('Invert', 0.6, None)),
                           (('ShearX', 0.6, 5), ('Equalize', 1.0, None)),
                           (('Color', 0.4, 0), ('Equalize', 0.6, None)),
                           (('Equalize', 0.4, None), ('Solarize', 0.2, 4)),
                           (('Solarize', 0.6, 5), ('AutoContrast', 0.6, None)),
                           (('Invert', 0.6, None), ('Equalize', 1.0, None)),
                           (('Color', 0.6, 4), ('Contrast', 1.0, 8)),
                           (('Equalize', 0.8, None), ('Equalize', 0.6, None))]

        return Policy(Policy.dda_operations(), 25, 0.05, 2, num_chunks=num_chunks, init_operations=init_operations, init_op_probability=init_op_probability)
    
    @staticmethod
    def init_w_weaken_auto_augment_policy(init_op_probability=0.9, num_chunks=4, div=2.0) -> Policy:
        init_operations = [(('Posterize', 0.4 / div, 8 / div), ('Rotate', 0.6 / div, 9 / div)),
                           (('Solarize', 0.6 / div, 5 / div), ('AutoContrast', 0.6 / div, None)),
                           (('Equalize', 0.8 / div, None), ('Equalize', 0.6 / div, None)),
                           (('Posterize', 0.6 / div, 7 / div), ('Posterize', 0.6 / div, 6 / div)),
                           (('Equalize', 0.4 / div, None), ('Solarize', 0.2 / div, 4 / div)),
                           (('Equalize', 0.4 / div, None), ('Rotate', 0.8 / div, 8 / div)),
                           (('Solarize', 0.6 / div, 3 / div), ('Equalize', 0.6 / div, None)),
                           (('Posterize', 0.8 / div, 5 / div), ('Equalize', 1.0 / div, None)),
                           (('Rotate', 0.2 / div, 3 / div), ('Solarize', 0.6 / div, 8 / div)),
                           (('Equalize', 0.6 / div, None), ('Posterize', 0.4 / div, 6 / div)),
                           (('Rotate', 0.8 / div, 8 / div), ('Color', 0.4 / div, 0 / div)),
                           (('Rotate', 0.4 / div, 9 / div), ('Equalize', 0.6 / div, None)),
                           (('Equalize', 0.0 / div, None), ('Equalize', 0.8 / div, None)),
                           (('Invert', 0.6 / div, None), ('Equalize', 1.0 / div, None)),
                           (('Color', 0.6 / div, 4 / div), ('Contrast', 1.0 / div, 8 / div)),
                           (('Rotate', 0.8 / div, 8 / div), ('Color', 1.0 / div, 2 / div)),
                           (('Color', 0.8 / div, 8 / div), ('Solarize', 0.8 / div, 7 / div)),
                           (('Sharpness', 0.4 / div, 7 / div), ('Invert', 0.6 / div, None)),
                           (('ShearX', 0.6 / div, 5 / div), ('Equalize', 1.0 / div, None)),
                           (('Color', 0.4 / div, 0 / div), ('Equalize', 0.6 / div, None)),
                           (('Equalize', 0.4 / div, None), ('Solarize', 0.2 / div, 4 / div)),
                           (('Solarize', 0.6 / div, 5 / div), ('AutoContrast', 0.6 / div, None)),
                           (('Invert', 0.6 / div, None), ('Equalize', 1.0 / div, None)),
                           (('Color', 0.6 / div, 4 / div), ('Contrast', 1.0 / div, 8 / div)),
                           (('Equalize', 0.8 / div, None), ('Equalize', 0.6 / div, None))]

        return Policy(Policy.dda_operations(), 25, 0.05, 2, num_chunks=num_chunks, init_operations=init_operations, init_op_probability=init_op_probability)
    
    @staticmethod
    def faster_auto_augment_policy_w_prob_and_mag(num_sub_policies: int,
                                                  temperature: float,
                                                  operation_count: int,
                                                  num_chunks: int,
                                                  init_prob: float=0.1, init_mag: float=0.1) -> Policy:

        return Policy(Policy.dda_operations(), num_sub_policies, temperature, operation_count, num_chunks, init_prob_and_mag=[init_prob, init_mag])
    
    @staticmethod
    def faster_auto_augment_policy_wo_negative(num_sub_policies: int,
                                              temperature: float,
                                              operation_count: int,
                                              num_chunks: int,
                                              init_prob: float=0.1, init_mag: float=0.1) -> Policy:

        return Policy(Policy.dda_operations_wo_negative(), num_sub_policies, temperature, operation_count, num_chunks, init_prob_and_mag=[init_prob, init_mag])