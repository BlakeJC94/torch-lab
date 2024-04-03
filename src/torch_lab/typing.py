from typing import TypeAlias
from numpy import ndarray
from torch import Tensor

CollateData: TypeAlias = int | float | str | ndarray | Tensor
