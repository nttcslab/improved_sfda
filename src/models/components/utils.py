from typing import Any, List, Optional

import numpy as np
import torch

def Entropy(input_):
    bs = input_.size(0)
    entropy = -input_ * torch.log(input_ + 1e-5)
    entropy = torch.sum(entropy, dim=1)
    return entropy

class WeightScheduler():
    def __init__(self, alpha: Optional[float] = 1.0, beta: Optional[float] = 1.0, max_iters: Optional[int] = 1000, **kwargs):
        super(WeightScheduler, self).__init__()

        self.alpha = alpha
        self.beta = beta
        self.iter_num = 0
        self.max_iters = max_iters

    def calculate(self) -> float:
        return (1 + 10 * self.iter_num / self.max_iters) ** (-self.beta) * self.alpha

    def step(self):
        self.iter_num += 1

class RampUpScheduler():
    def __init__(self, gamma=10, max_iters=1000):
        self.gamma = gamma
        self.max_iters = max_iters
        self.iter_num = 0

    def calculate(self) -> float:
        return 2.0 / (1.0 + np.exp(self.gamma * self.iter_num / self.max_iters)) - 1.0
    
    def step(self):
        self.iter_num += 1