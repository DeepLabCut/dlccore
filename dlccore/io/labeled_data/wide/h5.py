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

"""Labeled-data HDF5 files: which key to read, and the read itself.

Returns the shared ``table.LabeledDataRead``; classification is the same for every container. Imported explicitly,
so pandas and PyTables load only when it is used.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import pandas as pd

from dlccore.io.errors import UnreadableFileError
from dlccore.io.labeled_data.wide.table import LabeledDataRead
from dlccore.io.stamp import WRITER_KEY, WRITER_VERSION_KEY, WriterStamp

# Observed from a payload pandas cannot rebuild: TypeError (unknown storer), NoSuchNodeError
# (LookupError, AttributeError) for a missing child node, HDF5ExtError (RuntimeError), OSError.
_PAYLOAD_READ_ERRORS = (OSError, RuntimeError, TypeError, ValueError, LookupError, AttributeError)


class H5KeyStatus(str, Enum):
    """Confidence level for a key found in a labeled-data HDF5 file."""

    CURRENT = "current"
    LEGACY = "legacy"
    UNKNOWN = "unknown"


# The key DeepLabCut and napari-deeplabcut write today, and the one this package writes.
CURRENT_H5_KEY = "df_with_missing"

H5_KEY_REGISTRY: dict[str, H5KeyStatus] = {
    CURRENT_H5_KEY: H5KeyStatus.CURRENT,
    "keypoints": H5KeyStatus.LEGACY,
}


H5_KEY_PRECEDENCE: tuple[H5KeyStatus, ...] = (
    H5KeyStatus.CURRENT,
    H5KeyStatus.LEGACY,
    H5KeyStatus.UNKNOWN,
)


class H5KeySelectionOutcome(str, Enum):
    """Closed set of key-selection results, describing the file rather than our reaction.

    Every outcome but ``EMPTY`` names a readable file: DeepLabCut and
    napari-deeplabcut both read any single key, so refusing one is a
    backwards-compatibility defect here.
    """

    SINGLE_CURRENT = "single_current"
    SINGLE_LEGACY = "single_legacy"
    SINGLE_UNKNOWN = "single_unknown"
    MULTI_KEY = "multi_key"
    EMPTY = "empty"


_SINGLE_KEY_OUTCOMES: dict[H5KeyStatus, H5KeySelectionOutcome] = {
    H5KeyStatus.CURRENT: H5KeySelectionOutcome.SINGLE_CURRENT,
    H5KeyStatus.LEGACY: H5KeySelectionOutcome.SINGLE_LEGACY,
    H5KeyStatus.UNKNOWN: H5KeySelectionOutcome.SINGLE_UNKNOWN,
}


@dataclass(frozen=True)
class H5KeySelection:
    """Result of enumerating and selecting among a file's HDF5 keys.

    Attributes:
        keys_found: Keys pandas lists, in the file's order, without the leading ``/``.
        selected_key: The key read, ``None`` exactly when ``outcome`` is ``EMPTY``.
        outcome: What the keys found say about the file.
    """

    keys_found: tuple[str, ...]
    selected_key: str | None
    outcome: H5KeySelectionOutcome

    def __post_init__(self) -> None:
        """Refuse a selection that names a key for an empty file, or none for a non-empty one.

        Raises:
            ValueError: If ``selected_key`` is ``None`` and ``outcome`` is not ``EMPTY``, or the reverse.
        """

        if (self.selected_key is None) != (self.outcome is H5KeySelectionOutcome.EMPTY):
            raise ValueError("An H5KeySelection selects no key exactly when its outcome is EMPTY.")


def select_by_precedence(
    keys_found: tuple[str, ...],
    statuses: dict[str, H5KeyStatus],
) -> str:
    """Return the highest-precedence key; within one status, the file's own order wins.

    Declared rather than incidental: selecting among several keys without a
    stated order is the silent first-match the key contract forbids.

    Raises:
        ValueError: If no key carries a status listed in the precedence order.
    """

    for status in H5_KEY_PRECEDENCE:
        for key in keys_found:
            if statuses[key] is status:
                return key

    raise ValueError(f"No key among {keys_found} carries a status in {H5_KEY_PRECEDENCE}.")


def select_h5_key(path: Path) -> H5KeySelection:
    """Enumerate the pandas keys in an HDF5 file and select one per the key-selection policy.

    A single key is selected whatever its name; several keys select by declared precedence.

    Raises:
        UnreadableFileError: If the file cannot be opened as HDF5.
    """

    try:
        with pd.HDFStore(str(path), mode="r") as store:
            keys_found = tuple(key.lstrip("/") for key in store)
    except (OSError, RuntimeError) as error:
        raise UnreadableFileError(f"{path}: cannot be opened as HDF5: {error}") from error

    if not keys_found:
        return H5KeySelection(keys_found=keys_found, selected_key=None, outcome=H5KeySelectionOutcome.EMPTY)

    statuses = {key: H5_KEY_REGISTRY.get(key, H5KeyStatus.UNKNOWN) for key in keys_found}

    if len(keys_found) == 1:
        return H5KeySelection(
            keys_found=keys_found,
            selected_key=keys_found[0],
            outcome=_SINGLE_KEY_OUTCOMES[statuses[keys_found[0]]],
        )

    return H5KeySelection(
        keys_found=keys_found,
        selected_key=select_by_precedence(keys_found, statuses),
        outcome=H5KeySelectionOutcome.MULTI_KEY,
    )


def read_labeled_data_h5(path: Path) -> LabeledDataRead:
    """Read the selected key of a labeled-data HDF5 file as stored.

    Args:
        path: A ``CollectedData_*.h5`` or ``machinelabels*.h5`` file.

    Raises:
        UnreadableFileError: If the file cannot be opened, holds no pandas key, or its selected key does not
            read back as a DataFrame.
    """

    key_selection = select_h5_key(path)
    key = key_selection.selected_key

    if key is None:
        raise UnreadableFileError(f"{path}: holds no pandas key, so there is nothing to read.")

    try:
        payload = pd.read_hdf(path, key=key, mode="r")
    except _PAYLOAD_READ_ERRORS as error:
        raise UnreadableFileError(f"{path}: key {key!r} cannot be read: {error}") from error

    if not isinstance(payload, pd.DataFrame):
        raise UnreadableFileError(f"{path}: key {key!r} holds a {type(payload).__name__}, not a DataFrame.")

    stamp, stamp_error = _writer_stamp(path, key)
    return LabeledDataRead(
        path=path,
        dataframe=payload,
        key_selection=key_selection,
        writer_stamp=stamp,
        writer_stamp_error=stamp_error,
    )


def write_labeled_data_h5(df: pd.DataFrame, path: Path, stamp: WriterStamp) -> None:
    """Write a wide table whole under the current key, and stamp its node with who wrote it.

    The stamp is stored as plain string attributes, which HDF5 holds natively. Never appends: ``path`` is created
    or overwritten. Callers write to a temporary path and replace.
    """

    df.to_hdf(path, key=CURRENT_H5_KEY, mode="w")
    with pd.HDFStore(str(path), mode="a") as store:
        attributes = store.get_storer(CURRENT_H5_KEY).attrs
        for name, value in stamp.as_metadata().items():
            setattr(attributes, name, value)


def _writer_stamp(path: Path, key: str) -> tuple[WriterStamp | None, str | None]:
    """Return the writer stamp on ``key``'s node, or why a stamp that is present is malformed.

    A malformed stamp leaves the table readable, as it is to pandas and DeepLabCut.

    Raises:
        UnreadableFileError: If the attributes cannot be read.
    """

    try:
        with pd.HDFStore(str(path), mode="r") as store:
            attributes = store.get_storer(key).attrs
            names = set(attributes._v_attrnamesuser)
            metadata = {name: getattr(attributes, name) for name in (WRITER_KEY, WRITER_VERSION_KEY) if name in names}
    except _PAYLOAD_READ_ERRORS as error:
        raise UnreadableFileError(f"{path}: key {key!r} attributes cannot be read: {error}") from error
    try:
        return WriterStamp.from_metadata(metadata), None
    except ValueError as error:
        return None, f"{path}: key {key!r}: {error}"
