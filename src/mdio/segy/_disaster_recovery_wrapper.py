"""Consumer-side utility to get both raw and transformed header data with single filesystem read."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from segy import SegyFile


class SegyFileTraceDataWrapper:
    def __init__(self, segy_file: SegyFile, indices: int | list[int] | NDArray | slice):
        self.segy_file = segy_file
        self.indices = indices
        self._header_pipeline = deepcopy(segy_file.accessors.header_decode_pipeline)
        segy_file.accessors.header_decode_pipeline.transforms = []
        self.traces = segy_file.trace[indices]

    @property
    def header(self) -> NDArray:
        # The copy is necessary to avoid applying the pipeline to the original header.
        return self._header_pipeline.apply(self.traces.header.copy())

    @property
    def raw_header(self) -> NDArray:
        return np.ascontiguousarray(self.traces.header.copy()).view("|V240")

    @property
    def sample(self) -> NDArray:
        return self.traces.sample
