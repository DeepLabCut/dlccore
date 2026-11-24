"""``LabeledData`` as read-only arrays: axis 0 is the frame, axis 1 the keypoint."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from dlc_octopus.io.labeled_data.long.model import FRAME_COLUMNS, keypoints_of
from dlc_octopus.io.labeled_data.model import LIKELIHOOD_COORD, POSITION_COORDS, FramePath, Keypoint

if TYPE_CHECKING:
    from dlc_octopus.io.labeled_data.long.model import LabeledData


@dataclass(frozen=True, eq=False)
class LabeledDataArrays:
    """Every keypoint in every frame, as arrays: axis 0 is the frame, axis 1 the keypoint.

    Values are as stored, with no interpretation beyond "a coordinate pair is present". Compared and hashed by
    identity, since arrays have no single truth value.

    Attributes:
        frames: The frames, in frame order.
        keypoints: The keypoints, in keypoint order.
        xy: Shape ``(frames, keypoints, 2)``; values as stored, NaN where missing.
        likelihood: Shape ``(frames, keypoints)``, NaN where missing; ``None`` when the labels store none.
    """

    frames: tuple[FramePath, ...]
    keypoints: tuple[Keypoint, ...]
    xy: np.ndarray
    likelihood: np.ndarray | None

    def __post_init__(self) -> None:
        """Refuse arrays that do not have one entry per frame and keypoint.

        Raises:
            ValueError: If ``xy`` is not ``(frames, keypoints, 2)``, or ``likelihood`` not ``(frames, keypoints)``.
        """

        shape = (len(self.frames), len(self.keypoints))
        if self.xy.shape != (*shape, 2):
            raise ValueError(
                f"xy has shape {self.xy.shape}; {shape[0]} frames and {shape[1]} keypoints need {(*shape, 2)}."
            )
        if self.likelihood is not None and self.likelihood.shape != shape:
            raise ValueError(
                f"likelihood has shape {self.likelihood.shape}; {shape[0]} frames and {shape[1]} keypoints need "
                f"{shape}."
            )

    @property
    def labeled(self) -> np.ndarray:
        """Where both coordinates are present, shape ``(frames, keypoints)``; their values are not judged."""

        return ~np.isnan(self.xy).any(axis=-1)


def long_to_arrays(labels: LabeledData) -> LabeledDataArrays:
    """Return the labels as read-only arrays in frame then keypoint order, values as stored.

    Args:
        labels: The labels to view.
    """

    long = labels.to_long_df()
    n_frames, n_keypoints = long["frame_index"].nunique(), long["keypoint_index"].nunique()

    # Rows are frame-major, so the first frame's rows name the keypoints and every K-th row starts a frame.
    first_frame = long.iloc[:n_keypoints]
    frame_starts = long.iloc[::n_keypoints] if n_keypoints else long.iloc[:0]

    frames = tuple(zip(*(frame_starts[name].astype(str) for name in FRAME_COLUMNS), strict=True))
    keypoints = keypoints_of(first_frame)

    def values(coordinate: str) -> np.ndarray:
        return long[coordinate].to_numpy(dtype=float, copy=True).reshape(n_frames, n_keypoints)

    xy = np.stack([values(coordinate) for coordinate in POSITION_COORDS], axis=-1)
    likelihood = values(LIKELIHOOD_COORD) if labels.has_likelihood else None

    return LabeledDataArrays(frames, keypoints, _read_only(xy), None if likelihood is None else _read_only(likelihood))


def _read_only(array: np.ndarray) -> np.ndarray:
    array.setflags(write=False)
    return array
