"""Custom types used in torch_lab."""
from typing import Any, Dict, List, Optional, TypeAlias

from numpy import ndarray
from torch import Tensor

CollateData: TypeAlias = int | float | str | ndarray | Tensor
JsonDict: TypeAlias = Dict[str, Optional[str | float | int | List[Any] | Dict[str, Any]]]
