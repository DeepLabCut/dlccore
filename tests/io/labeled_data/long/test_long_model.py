import re
from collections.abc import Callable
from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest

from dlc_octopus.io.labeled_data.long.model import LABELED_DATA_LONG_VERSION, LabeledData

TEXT_COLUMNS = ("folder", "dataset", "image", "scorer", "individual", "bodypart")
# What each long-format version holds, multi-animal with likelihood: columns in order, with their dtypes.
LONG_FORMATS = {
    "0.0": [
        *((column, "category") for column in TEXT_COLUMNS),
        ("x", "float64"),
        ("y", "float64"),
        ("likelihood", "float64"),
        ("frame_index", "int64"),
        ("keypoint_index", "int64"),
    ],
}


def _long(*, multi: bool = False, likelihood: bool = False) -> pd.DataFrame:
    """Two frames by two keypoints, frame-major, with text columns categorical as the contract requires."""

    rows = []
    for frame_index, image in enumerate(("img0.png", "img1.png")):
        for keypoint_index, bodypart in enumerate(("nose", "tail")):
            row = {
                "folder": "labeled-data",
                "dataset": "s1",
                "image": image,
                "scorer": "Sc",
                "individual": "mouse1",
                "bodypart": bodypart,
                "x": 1.0 + frame_index,
                "y": np.nan,
                "likelihood": 0.5,
                "frame_index": frame_index,
                "keypoint_index": keypoint_index,
            }
            rows.append(row)
    df = pd.DataFrame(rows)
    if not multi:
        df = df.drop(columns="individual")
    if not likelihood:
        df = df.drop(columns="likelihood")
    for column in TEXT_COLUMNS:
        if column in df.columns:
            df[column] = df[column].astype("category")
    return df


@pytest.mark.parametrize("multi", [False, True], ids=["single", "multi"])
@pytest.mark.parametrize("likelihood", [False, True], ids=["xy", "likelihood"])
def test_a_table_holding_the_contract_is_wrapped_unchanged(*, multi: bool, likelihood: bool) -> None:
    # Control for every refusal below.
    df = _long(multi=multi, likelihood=likelihood)

    labels = LabeledData.from_long_df(df)

    assert labels.is_multi_animal is multi
    assert labels.has_likelihood is likelihood
    pd.testing.assert_frame_equal(labels.to_long_df(), df[labels.to_long_df().columns])


def test_rows_in_any_order_come_back_ordered_by_frame_then_keypoint() -> None:
    df = _long()

    labels = LabeledData.from_long_df(df.iloc[::-1])

    pd.testing.assert_frame_equal(labels.to_long_df(), df[labels.to_long_df().columns])


def test_the_table_cannot_be_changed_through_its_view_or_its_attributes() -> None:
    labels = LabeledData.from_long_df(_long())

    view = labels.to_long_df()
    view.loc[0, "x"] = 99.0

    assert labels.to_long_df().loc[0, "x"] == 1.0
    with pytest.raises(FrozenInstanceError):
        labels._frame = _long()  # type: ignore[misc]


def test_constructing_directly_checks_the_contract_too() -> None:
    with pytest.raises(ValueError, match="Columns"):
        LabeledData(_long().drop(columns="bodypart"))


def test_categories_are_canonical_whatever_the_caller_built() -> None:
    # One dataset, one representation: unused categories dropped, the rest sorted.
    canonical = LabeledData.from_long_df(_long()).to_long_df()
    df = _long()
    df["bodypart"] = df["bodypart"].cat.add_categories(["ghost"]).cat.reorder_categories(["tail", "ghost", "nose"])

    built = LabeledData.from_long_df(df).to_long_df()

    pd.testing.assert_frame_equal(built, canonical)
    assert list(built["bodypart"].cat.categories) == ["nose", "tail"]


def _drop_column(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns="bodypart")


def _add_column(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(scope="individual")


def _text_as_object(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(image=df["image"].astype(object))


def _numeric_categories(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(dataset=pd.Categorical([1, 1, 1, 1]))


def _missing_text(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(bodypart=df["bodypart"].where(df.index != 0))


def _integer_values(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(x=df["x"].astype("int64"))


def _float_order(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(frame_index=df["frame_index"].astype("float64"))


def _no_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df.iloc[:0]


def _row_missing(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(index=3)


def _row_repeated(df: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([df, df.iloc[[0]]])


def _index_skipped(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(frame_index=df["frame_index"] * 2)


def _frame_parts_disagree(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(image=pd.Categorical(["img0.png", "other.png", "img1.png", "img1.png"]))


def _keypoint_names_disagree(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(bodypart=pd.Categorical(["nose", "tail", "ear", "tail"]))


def _two_indices_one_keypoint(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(bodypart=pd.Categorical(["nose"] * 4))


@pytest.mark.parametrize(
    ("break_contract", "message"),
    [
        (_drop_column, "Columns"),
        (_add_column, "Columns"),
        (_text_as_object, "categorical"),
        (_numeric_categories, "categorical"),
        (_missing_text, "missing at row 0"),
        (_integer_values, "float64"),
        (_float_order, "int64"),
        (_no_rows, "no frames"),
        (_row_missing, re.escape("missing (1, 1)")),
        (_row_repeated, re.escape("repeated (0, 0)")),
        (_index_skipped, re.escape("missing (1, 0); out of range (2, 0)")),
        (_frame_parts_disagree, re.escape("frame_index 0 disagree on ['image']")),
        (_keypoint_names_disagree, re.escape("keypoint_index 0 disagree on ['bodypart']")),
        (_two_indices_one_keypoint, re.escape("values [0, 1] all name the keypoint ('Sc', 'nose')")),
    ],
)
def test_a_table_breaking_the_contract_is_refused_with_the_rule(
    break_contract: Callable[[pd.DataFrame], pd.DataFrame], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        LabeledData.from_long_df(break_contract(_long()))


def test_the_long_format_changes_only_with_its_version() -> None:
    held = LabeledData.from_long_df(_long(multi=True, likelihood=True)).to_long_df()

    assert [(column, str(dtype)) for column, dtype in held.dtypes.items()] == LONG_FORMATS[LABELED_DATA_LONG_VERSION]
