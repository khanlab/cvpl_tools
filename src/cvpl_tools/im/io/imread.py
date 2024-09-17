"""
This file defines how to read images whose representations are n-dimensional arrays, occasionally with metadata
associated with them.
"""

from abc import ABC, abstractmethod
from enum import Enum, auto

import numpy as np
import numpy.typing as npt
import dask.array as da


class ImageFileTypes(Enum):
    PNG = auto()
    TIF = auto()
    NIFTI = auto()
    ZARR = auto()
    OME_ZARR = auto()


class NumpyInterface(ABC):

    @abstractmethod
    def get_numpy(self) -> npt.NDArray:
        raise NotImplementedError()

    @abstractmethod
    def set_numpy(self, arr: npt.NDArray):
        raise NotImplementedError()

    @abstractmethod
    @property
    def dtype(self) -> np.dtype:
        raise NotImplementedError()

    @abstractmethod
    @property
    def ndim(self) -> int:
        raise NotImplementedError()

    @abstractmethod
    @property
    def shape(self) -> tuple[int, ...]:
        raise NotImplementedError()


class DaskInterface(ABC):
    pass
