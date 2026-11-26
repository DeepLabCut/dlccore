import pytest
from pydantic import BaseModel

from dlc_octopus.capabilities import get_project_manager_capabilities
from dlc_octopus.schemas.base.filesystem import walk_schema
from dlc_octopus.schemas.base.resources import DirectoryResourceKind, FileResourceKind, ResourceManagement
from dlc_octopus.schemas.base.validation import ValidationDepth
from dlc_octopus.schemas.capabilities import (
    BaselineValidationCapability,
    LayoutCapability,
    ProjectManagerCapabilities,
    ValidationCapability,
)
from dlc_octopus.schemas.project_layout.project import PROJECT
from dlc_octopus.visualization import render_capabilities


def _layout(schema_id: str, *children: LayoutCapability, **fields: object) -> LayoutCapability:
    return LayoutCapability(
        schema_id=schema_id,
        kind=fields.pop("kind", DirectoryResourceKind.PROJECT_ROOT),
        required=fields.pop("required", False),
        management=fields.pop("management", ResourceManagement.MANAGED),
        children=children,
        **fields,
    )


def test_every_layout_node_and_check_of_this_installation_is_rendered() -> None:
    capabilities = get_project_manager_capabilities()
    rendered = render_capabilities(capabilities)

    for schema, _ in walk_schema(PROJECT):
        assert f" {schema.schema_id} [" in rendered
    for check in capabilities.baseline_validations:
        assert check.code in rendered
    for validation in capabilities.validations:
        assert f"  {validation.kind.value}: " in rendered


def test_the_layout_is_a_tree_with_each_node_flagged() -> None:
    capabilities = ProjectManagerCapabilities(
        snapshot_version="9.9",
        layout=_layout(
            "dlc.t",
            _layout("dlc.t.file", kind=FileResourceKind.PROJECT_CONFIG, required=True),
            _layout(
                "dlc.t.dir",
                _layout(
                    "dlc.t.dir.item",
                    kind=FileResourceKind.SOURCE_VIDEO,
                    repeated=True,
                    management=ResourceManagement.KNOWN_UNMANAGED,
                ),
            ),
            required=True,
        ),
    )

    rendered = render_capabilities(capabilities).splitlines()

    assert rendered[0] == "Octopus capabilities (snapshot version 9.9)"
    assert rendered[3:7] == [
        "└── dlc.t [project_root, required]",
        "    ├── dlc.t.file [project_config, required]",
        "    └── dlc.t.dir [project_root, optional]",
        "        └── dlc.t.dir.item [source_video, optional, repeated, known_unmanaged]",
    ]


def test_checks_are_aligned_and_no_validators_is_said_so() -> None:
    capabilities = ProjectManagerCapabilities(
        snapshot_version="9.9",
        layout=_layout("dlc.t"),
        baseline_validations=(
            BaselineValidationCapability(code="SHORT", description="first"),
            BaselineValidationCapability(code="MUCH_LONGER", description="second"),
        ),
    )

    rendered = render_capabilities(capabilities)

    assert "  SHORT        first\n  MUCH_LONGER  second" in rendered
    assert rendered.endswith("Optional validators, on request\n  none")


def _flagged() -> ProjectManagerCapabilities:
    return ProjectManagerCapabilities(
        snapshot_version="9.9",
        layout=_layout(
            "dlc.t",
            _layout("dlc.t.item", kind=FileResourceKind.SOURCE_VIDEO, repeated=True),
            required=True,
        ),
        baseline_validations=(BaselineValidationCapability(code="A_CHECK", description="one | two"),),
        validations=(
            ValidationCapability(
                kind=DirectoryResourceKind.VIDEOS_DIRECTORY,
                supported_depths=(ValidationDepth.FILENAME_ONLY, ValidationDepth.FULL),
            ),
        ),
    )


def test_markdown_lists_the_layout_and_tabulates_checks_and_validators() -> None:
    rendered = render_capabilities(_flagged(), style="markdown").splitlines()

    assert rendered[0] == "**Octopus capabilities** (snapshot version `9.9`)"
    assert "- `dlc.t`: project_root, required" in rendered
    assert "  - `dlc.t.item`: source_video, optional, repeated" in rendered
    assert "| `A_CHECK` | one \| two |" in rendered
    assert "| `videos_directory` | filename_only, full |" in rendered


def test_markdown_says_when_there_is_nothing_to_tabulate() -> None:
    rendered = render_capabilities(
        ProjectManagerCapabilities(snapshot_version="9.9", layout=_layout("dlc.t")), style="markdown"
    )

    assert rendered.count("None.") == 2
    assert "|---|" not in rendered


def test_an_unknown_style_is_refused() -> None:
    with pytest.raises(ValueError, match="'html'"):
        render_capabilities(_flagged(), style="html")  # type: ignore[arg-type]


def test_capabilities_display_themselves_and_keep_pydantics_repr() -> None:
    capabilities = _flagged()
    pretty = pytest.importorskip("IPython.lib.pretty").pretty

    assert (type(capabilities).__repr__, type(capabilities).__str__) == (BaseModel.__repr__, BaseModel.__str__)
    assert capabilities._repr_markdown_() == render_capabilities(capabilities, style="markdown")
    assert pretty(capabilities) == render_capabilities(capabilities)
