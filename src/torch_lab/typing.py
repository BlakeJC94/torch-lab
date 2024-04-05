from __future__ import annotations

from typing import TypeAlias, Dict, Optional, List
from numpy import ndarray
from torch import Tensor

CollateData: TypeAlias = int | float | str | ndarray | Tensor
JsonDict: TypeAlias = Dict[str, Optional[str | float | int | List[JsonDict] | Dict[str, JsonDict]]]
