from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import BaseModel

from dlc_octopus.schemas.base.finding_codes import ConfigFindingCode, StructureFindingCode, iter_finding_codes
from dlc_octopus.schemas.base.findings import Finding, Reach
from dlc_octopus.visualization import render_finding, render_finding_catalogue, render_findings_markdown

NO_KEYPOINTS = "The config loads but declares no keypoints."
SEP = chr(0x203A)  # between a file and a field in Markdown


def _no_keypoints(root: Path) -> Finding:
    return Finding(
        code=ConfigFindingCode.CONFIG_NO_KEYPOINTS,
        schema_id="dlc.project.config",
        reach=Reach.ONE_PER_PROJECT,
        path=root / "config.yaml",
        field_location=("bodyparts",),
        evidence={"multianimal_project": False, "declared": {"bodyparts": []}},
    )


def _moved(root: Path) -> Finding:
    return Finding(
        code=ConfigFindingCode.CONFIG_PROJECT_PATH_MISMATCH,
        schema_id="dlc.project.config",
        reach=Reach.ONE_PER_PROJECT,
        path=root / "config.yaml",
        field_location=("project_path",),
        evidence={
            "declared_project_path": "/old/place",
            "observed_project_root": str(root),
            "declared_path_probe": "missing",
            "error_type": None,
            "error_message": None,
        },
    )


def _unreadable(instance: tuple[str, ...], **evidence: str) -> Finding:
    code = StructureFindingCode.PROJECT_NODE_UNREADABLE
    return Finding(
        code=code,
        schema_id="dlc.labeled_data.dataset",
        reach=Reach.ONE_OF,
        instance=instance,
        evidence={**dict.fromkeys(code.evidence_keys, "x"), **evidence},
    )


def test_text_puts_the_location_first_then_the_description_and_code_then_evidence(tmp_path: Path) -> None:
    assert render_finding(_no_keypoints(tmp_path), project_root=tmp_path) == (
        "WARNING  config.yaml:bodyparts\n"
        f"         {NO_KEYPOINTS}  [CONFIG_NO_KEYPOINTS]\n"
        "         multianimal_project: False\n"
        "         declared: {'bodyparts': []}"
    )


def test_compact_is_one_line(tmp_path: Path) -> None:
    assert render_finding(_no_keypoints(tmp_path), project_root=tmp_path, style="compact") == (
        f"WARNING CONFIG_NO_KEYPOINTS config.yaml:bodyparts: {NO_KEYPOINTS} "
        "(multianimal_project=False, declared={'bodyparts': []})"
    )


def test_markdown_renders_one_finding(tmp_path: Path) -> None:
    assert render_finding(_no_keypoints(tmp_path), project_root=tmp_path, style="markdown") == (
        f"**WARNING** `config.yaml` {SEP} `bodyparts`: {NO_KEYPOINTS} `CONFIG_NO_KEYPOINTS`\n"
        "- `multianimal_project`: `False`\n"
        "- `declared`: `{'bodyparts': []}`"
    )


def test_the_location_falls_back_from_path_to_instance_to_layout_node(tmp_path: Path) -> None:
    outside = _no_keypoints(Path("/elsewhere"))
    nowhere = Finding(
        code=ConfigFindingCode.CONFIG_NO_KEYPOINTS,
        schema_id="dlc.project.config",
        reach=Reach.ONE_PER_PROJECT,
        evidence={"multianimal_project": False, "declared": {}},
    )

    def first_line(finding: Finding) -> str:
        return render_finding(finding, project_root=tmp_path, style="compact").split(": ")[0]

    assert first_line(outside) == f"WARNING CONFIG_NO_KEYPOINTS {Path('/elsewhere/config.yaml')}:bodyparts"
    assert first_line(_unreadable(("labeled-data", "mouse1"))).endswith(" labeled-data/mouse1")
    assert first_line(nowhere) == "WARNING CONFIG_NO_KEYPOINTS dlc.project.config"


def test_an_unknown_style_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown finding style"):
        render_finding(_no_keypoints(tmp_path), style="html")  # type: ignore[arg-type]


def test_a_markdown_list_tallies_severities_and_folds_the_evidence(tmp_path: Path) -> None:
    rendered = render_findings_markdown([_no_keypoints(tmp_path), _moved(tmp_path)], project_root=tmp_path)

    assert rendered.splitlines()[:5] == [
        "**2 findings** · 1 warning · 1 note",
        "",
        "| Severity | Where | What | Code |",
        "|---|---|---|---|",
        f"| WARNING | `config.yaml` {SEP} `bodyparts` | {NO_KEYPOINTS} | `CONFIG_NO_KEYPOINTS` |",
    ]
    assert "<details><summary>Evidence</summary>" in rendered
    assert render_findings_markdown([]) == "No findings."


def test_a_markdown_list_redacts_the_project_root_unless_asked_not_to(tmp_path: Path) -> None:
    redacted = render_findings_markdown([_moved(tmp_path)], project_root=tmp_path)
    kept = render_findings_markdown([_moved(tmp_path)], project_root=tmp_path, redact_root=False)

    assert str(tmp_path) not in redacted
    assert "`observed_project_root`: `<project>`" in redacted
    assert str(tmp_path) in kept  # control: the root is in the evidence to begin with


def test_markdown_escapes_what_would_break_a_table_or_a_code_span() -> None:
    finding = _unreadable(("labeled-data", "a|b"), error_message="unexpected `token`")

    rendered = render_findings_markdown([finding])

    assert "| `labeled-data/a\\|b` |" in rendered
    assert "`` unexpected `token` ``" in rendered


def test_repr_str_and_serialization_stay_pydantics(tmp_path: Path) -> None:
    finding = _no_keypoints(tmp_path)

    assert (type(finding).__repr__, type(finding).__str__) == (BaseModel.__repr__, BaseModel.__str__)
    assert repr(finding).startswith("Finding(code=")
    assert Finding.model_validate_json(finding.model_dump_json()) == finding


def test_jupyter_shows_a_finding_as_markdown(tmp_path: Path) -> None:
    finding = _no_keypoints(tmp_path)

    assert finding._repr_markdown_() == render_finding(finding, style="markdown")


def test_ipython_pretty_prints_findings_alone_and_inside_containers(tmp_path: Path) -> None:
    pretty = pytest.importorskip("IPython.lib.pretty").pretty
    finding = _no_keypoints(tmp_path)

    assert pretty(finding) == render_finding(finding)
    assert render_finding(finding).splitlines()[1].strip() in pretty((finding, _moved(tmp_path)))


def test_the_catalogue_lists_every_code_once_under_its_category() -> None:
    codes = iter_finding_codes()
    categories = list(dict.fromkeys(type(code) for code in codes))
    rendered = render_finding_catalogue()
    sections = rendered.split("\n\n")

    assert [section.split(":")[0] for section in sections] == [
        "Structure",
        "Config",
        "Configured video",
        "Fingerprint",
        "Labeled data",
    ]
    for code in codes:
        assert rendered.count(f" {code.value} ") == 1
        assert f" {code.value} " in sections[categories.index(type(code))]


def test_the_markdown_catalogue_has_one_table_per_category() -> None:
    rendered = render_finding_catalogue(style="markdown")

    assert rendered.count("|---|---|---|") == len({type(code) for code in iter_finding_codes()})
    assert "**Configured video**: Findings about video paths declared in `config.yaml`." in rendered
    assert "``" not in rendered


def test_an_unknown_catalogue_style_is_refused() -> None:
    with pytest.raises(ValueError, match="'compact'"):
        render_finding_catalogue(style="compact")  # type: ignore[arg-type]
