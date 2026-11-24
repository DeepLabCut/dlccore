#
# DeepLabCut Toolbox (deeplabcut.org)
# © A. & M.W. Mathis Labs
# https://github.com/DeepLabCut/dlccore
#
# Contributors:
# https://github.com/DeepLabCut/dlccore/graphs/contributors
#
# Licensed under GNU Lesser General Public License v3.0
#

"""``LabeledData``: every keypoint in every frame as one long table, and the contract it holds."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import pandas as pd

from dlccore.io.labeled_data.model import LIKELIHOOD_COORD, POSITION_COORDS, Keypoint

if TYPE_CHECKING:
    from dlccore.io.labeled_data.arrays.from_long import LabeledDataArrays

FRAME_COLUMNS = ("folder", "dataset", "image")
SINGLE_ANIMAL_KEYPOINT_COLUMNS = ("scorer", "bodypart")
MULTI_ANIMAL_KEYPOINT_COLUMNS = ("scorer", "individual", "bodypart")
ORDER_COLUMNS = ("frame_index", "keypoint_index")

# 0.x carries no compatibility promise.
LabeledDataLongVersion = Literal["0.0"]
LABELED_DATA_LONG_VERSION: LabeledDataLongVersion = "0.0"


@dataclass(frozen=True, eq=False)
class LabeledData:
    """Every keypoint in every frame, one row each: dense, NaN where unlabeled, frozen.

    Strict on columns, dtypes and identities, never on values.
    Every method returns a new object, and the table itself is private.
    Compared and hashed by identity, as a DataFrame has neither value
    equality nor a hash.

    Methods convert only to forms built from core dependencies (the wide DataFrame, arrays). A format that needs an
    optional extra is converted by free functions in that format's own module, so this class never imports one.
    """

    _df: pd.DataFrame = field(repr=False)

    def __post_init__(self) -> None:
        """Check the table against the contract, keeping a copy in frame then keypoint order.

        Raises:
            ValueError: Naming the first broken rule: columns, dtypes, density, or identities.
        """

        object.__setattr__(self, "_df", _checked(self._df))

    @classmethod
    def from_long_df(cls, df: pd.DataFrame) -> LabeledData:
        """Wrap a long table holding the contract.

        Args:
            df: One row per frame and keypoint, in either schema; any column and row order.

        Raises:
            ValueError: Naming the first broken rule: columns, dtypes, density, or identities.
        """

        return cls(df)

    @classmethod
    def from_wide_df(cls, df: pd.DataFrame) -> LabeledData:
        """Interpret a wide table as a read does: columns by classified position, frame parts as text.

        Args:
            df: A table in any wide form DeepLabCut reads, such as one from ``LabeledDataFile.to_raw_wide_df()``.

        Raises:
            ValueError: Naming why the table has no model, as ``LabeledDataFile.labeled_data`` does.
        """

        from dlc_octopus.io.labeled_data.wide.to_long import wide_frame_to_long

        return wide_frame_to_long(df)

    @property
    def is_multi_animal(self) -> bool:
        return identity_columns(self._df) == MULTI_ANIMAL_KEYPOINT_COLUMNS

    @property
    def has_likelihood(self) -> bool:
        return LIKELIHOOD_COORD in self._df.columns

    def to_arrays(self) -> LabeledDataArrays:
        from dlccore.io.labeled_data.arrays.from_long import long_to_arrays

        return long_to_arrays(self)

    def to_wide_df(self) -> pd.DataFrame:
        from dlccore.io.labeled_data.wide.from_long import long_to_wide

        return long_to_wide(self)

    def to_long_df(self) -> pd.DataFrame:
        """A copy of the table, rows ordered by frame then keypoint."""
        return self._df.copy()


def identity_columns(df: pd.DataFrame) -> tuple[str, ...]:
    """The columns naming a keypoint: the only place that branches on single- or multi-animal."""

    return MULTI_ANIMAL_KEYPOINT_COLUMNS if "individual" in df.columns else SINGLE_ANIMAL_KEYPOINT_COLUMNS


def identity_values(keypoints: Sequence[Keypoint], *, multi_animal: bool) -> dict[str, list[str | None]]:
    """The identity columns of ``keypoints``, one value per keypoint: the only place attributes become columns."""

    values = {
        "scorer": [k.scorer for k in keypoints],
        "individual": [k.individual for k in keypoints],
        "bodypart": [k.bodypart for k in keypoints],
    }
    columns = MULTI_ANIMAL_KEYPOINT_COLUMNS if multi_animal else SINGLE_ANIMAL_KEYPOINT_COLUMNS
    return {name: values[name] for name in columns}


def keypoints_of(identities: pd.DataFrame) -> tuple[Keypoint, ...]:
    """One ``Keypoint`` per row of a table holding the identity columns, in row order."""

    columns = {name: identities[name].astype(str).tolist() for name in identity_columns(identities)}
    individuals = columns.get("individual", [None] * len(identities))
    return tuple(
        Keypoint(scorer=s, individual=i, bodypart=b)
        for s, i, b in zip(columns["scorer"], individuals, columns["bodypart"], strict=True)
    )


def _checked(df: pd.DataFrame) -> pd.DataFrame:
    keypoint_columns = identity_columns(df)
    text_columns = FRAME_COLUMNS + keypoint_columns
    value_columns = POSITION_COORDS + ((LIKELIHOOD_COORD,) if LIKELIHOOD_COORD in df.columns else ())
    expected = text_columns + value_columns + ORDER_COLUMNS

    if sorted(df.columns) != sorted(expected):
        raise ValueError(f"Columns are {sorted(df.columns)}; this schema needs exactly {sorted(expected)}.")

    for column in text_columns:
        dtype = df[column].dtype
        if not isinstance(dtype, pd.CategoricalDtype) or not all(isinstance(c, str) for c in dtype.categories):
            raise ValueError(f"Column {column!r} is {dtype}; it must be categorical with text categories.")
        if df[column].isna().any():
            row = df.index[df[column].isna()][0]
            raise ValueError(f"Column {column!r} is missing at row {row}; every row names its frame and keypoint.")
    for column in value_columns:
        if df[column].dtype != "float64":
            raise ValueError(f"Column {column!r} is {df[column].dtype}; it must be float64.")
    for column in ORDER_COLUMNS:
        if df[column].dtype != "int64":
            raise ValueError(f"Column {column!r} is {df[column].dtype}; it must be int64.")
    if df.empty:
        raise ValueError("The table has no frames; the model holds keypoints only through frames.")

    checked = df[list(expected)].sort_values(list(ORDER_COLUMNS)).reset_index(drop=True)
    _check_dense(checked)

    for index, identity in (("frame_index", FRAME_COLUMNS), ("keypoint_index", keypoint_columns)):
        disagreeing = checked.groupby(index, observed=True)[list(identity)].nunique() > 1
        if disagreeing.any(axis=None):
            value = disagreeing.any(axis=1).idxmax()
            columns = list(disagreeing.columns[disagreeing.loc[value]])
            raise ValueError(f"Rows with {index} {int(value)} disagree on {columns}.")

    keypoints = checked.drop_duplicates("keypoint_index")
    repeated = keypoints[keypoints.duplicated(list(keypoint_columns), keep=False)]
    if not repeated.empty:
        groups = repeated.groupby(list(keypoint_columns), observed=True)["keypoint_index"].apply(list)
        raise ValueError(f"keypoint_index values {groups.iloc[0]} all name the keypoint {groups.index[0]}.")

    # One representation per dataset: only the categories in use, in sorted order.
    for column in text_columns:
        used = checked[column].cat.remove_unused_categories()
        checked[column] = used.cat.reorder_categories(sorted(used.cat.categories))

    return checked


def _check_dense(df: pd.DataFrame) -> None:
    """Raise naming the first repeated, missing or out-of-range (frame_index, keypoint_index) pair, if any."""

    n_frames, n_keypoints = df["frame_index"].nunique(), df["keypoint_index"].nunique()
    dense = pd.MultiIndex.from_product([range(n_frames), range(n_keypoints)], names=list(ORDER_COLUMNS))
    actual = pd.MultiIndex.from_frame(df[list(ORDER_COLUMNS)])

    if actual.equals(dense):
        return

    offenders = (
        ("repeated", actual[actual.duplicated()]),
        ("missing", dense.difference(actual)),
        ("out of range", actual.difference(dense)),
    )
    first = "; ".join(f"{kind} {tuple(int(i) for i in pairs[0])}" for kind, pairs in offenders if len(pairs))
    raise ValueError(
        f"Rows must be dense, each frame_index 0..{n_frames - 1} with each keypoint_index 0..{n_keypoints - 1} exactly "
        f"once; first (frame_index, keypoint_index) {first}."
    )
