import random

from torch_lab.transforms import BaseDataTransform, BaseTransform


class RandomFlip(BaseTransform):
    def __init__(self, p: float):
        super().__init__()
        self.p = p

    def compute(self, x, md):
        if random.random() < self.p:
            x = x[..., ::-1, :]
        return x, md


class Scale(BaseDataTransform):
    def __init__(self, k: float):
        super().__init__()
        self.k = k

    def compute(self, x):
        return self.k * x
