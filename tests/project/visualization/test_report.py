from __future__ import annotations

from pathlib import Path

import pytest

from dlc_octopus.api import inspect_project
from dlc_octopus.schemas.base.finding_codes import StructureFindingCode
from dlc_octopus.schemas.base.findings import Finding, Reach
from dlc_octopus.schemas.base.resources import DirectoryResourceKind
from dlc_octopus.schemas.base.validation import ValidationDepth
from dlc_octopus.validation import ProjectValidationRequest, ResourceValidationRequest
from dlc_octopus.visualization import render_project_report


def test_render_project_report_shows_valid_project_summary(
    valid_project: Path,
) -> None:
    video_path = valid_project / "videos" / "session-01.mp4"
    video_path.touch()

    snapshot = inspect_project(valid_project)
    rendered = render_project_report(snapshot)

    assert "DeepLabCut project inspection" in rendered
    assert f"Project root: {valid_project}" in rendered

    assert "Configuration" in rendered
    assert "Status: valid" in rendered
    assert f"Config file: {valid_project / 'config.yaml'}" in rendered

    assert "Resource summary" in rendered
    assert "source_video: 1" in rendered

    assert "Project layout" in rendered
    assert "session-01.mp4" in rendered


def test_render_project_report_shows_missing_config_findings(
    tmp_path: Path,
) -> None:
    snapshot = inspect_project(tmp_path)
    rendered = render_project_report(snapshot)

    assert "Status: not found" in rendered
    assert "Findings" in rendered
    assert "PROJECT_REQUIRED_NODE_MISSING" in rendered


def test_render_project_report_shows_bare_finding_codes(
    tmp_path: Path,
) -> None:
    snapshot = inspect_project(tmp_path)
    rendered = render_project_report(snapshot)

    assert "- ERROR PROJECT_REQUIRED_NODE_MISSING" in rendered
    assert "StructureFindingCode" not in rendered


def _video_request() -> ProjectValidationRequest:
    return ProjectValidationRequest(
        resources=(
            ResourceValidationRequest(
                kind=DirectoryResourceKind.VIDEOS_DIRECTORY,
                depth=ValidationDepth.FILENAME_ONLY,
            ),
        )
    )


def test_render_project_report_says_when_no_resource_check_ran(
    valid_project: Path,
) -> None:
    rendered = render_project_report(inspect_project(valid_project))

    assert "Baseline structure and config checks always run and are not listed." in rendered
    assert "No resource checks were requested." in rendered


def test_render_project_report_can_omit_the_baseline_note(
    valid_project: Path,
) -> None:
    rendered = render_project_report(inspect_project(valid_project), show_baseline_note=False)

    assert "Baseline structure and config checks" not in rendered
    assert "Checks\n------\nNo resource checks were requested." in rendered


def test_render_project_report_lists_a_completed_check(
    valid_project: Path,
) -> None:
    rendered = render_project_report(inspect_project(valid_project, validation=_video_request()))

    assert "- VIDEO_FILENAME_VALIDATION on videos_directory (filename_only): completed" in rendered
    assert rendered.index("Checks\n------") < rendered.index("Findings\n--------")


def test_render_project_report_explains_an_unavailable_check(
    invalid_project: Path,
) -> None:
    snapshot = inspect_project(invalid_project, validation=_video_request())
    rendered = render_project_report(snapshot)

    (check,) = snapshot.validation_checks

    assert "- VIDEO_FILENAME_VALIDATION on videos_directory (filename_only): unavailable" in rendered
    assert f"    {check.message}" in rendered


def test_render_project_report_shows_each_findings_schema_id(
    tmp_path: Path,
) -> None:
    rendered = render_project_report(inspect_project(tmp_path))

    assert "- ERROR PROJECT_REQUIRED_NODE_MISSING at dlc.project.config [" in rendered


def test_render_project_report_shows_an_instance_after_its_schema_id(
    tmp_path: Path,
) -> None:
    code = StructureFindingCode.PROJECT_NODE_UNREADABLE
    finding = Finding(
        code=code,
        schema_id="dlc.labeled_data.dataset",
        reach=Reach.ONE_OF,
        instance=("labeled-data", "ds1"),
        evidence=dict.fromkeys(code.evidence_keys, "placeholder"),
    )
    snapshot = inspect_project(tmp_path).model_copy(update={"findings": (finding,)})

    rendered = render_project_report(snapshot)

    assert "- ERROR PROJECT_NODE_UNREADABLE at dlc.labeled_data.dataset (labeled-data/ds1): " in rendered


def test_render_project_report_shows_declared_evidence_in_order(
    tmp_path: Path,
) -> None:
    code = StructureFindingCode.PROJECT_NODE_WRONG_TYPE
    finding = Finding(
        code=code,
        schema_id="dlc.videos",
        reach=Reach.ONE_PER_PROJECT,
        evidence={"expected_type": "directory", "resource_kind": "videos_directory"},
    )
    snapshot = inspect_project(tmp_path).model_copy(update={"findings": (finding,)})

    rendered = render_project_report(snapshot)

    assert code.evidence_keys == ("resource_kind", "expected_type")
    assert rendered.index("    resource_kind=videos_directory") < rendered.index("    expected_type=directory")


def test_render_project_report_collapses_multiline_evidence(
    tmp_path: Path,
) -> None:
    (tmp_path / "config.yaml").write_text(
        "a:\n  b: [\n",
        encoding="utf-8",
    )

    rendered = render_project_report(inspect_project(tmp_path))
    message_lines = [line for line in rendered.splitlines() if line.startswith("    error_message=")]

    assert len(message_lines) == 1
    assert "error_type=" in rendered


def test_render_project_report_shows_failed_config(
    invalid_project: Path,
) -> None:
    snapshot = inspect_project(invalid_project)
    rendered = render_project_report(snapshot)

    assert "Status: failed" in rendered
    assert "Raw config retained for diagnostics: yes" in rendered
    assert "CONFIG_VALIDATION_ERROR" in rendered


def test_render_project_report_shows_no_findings_message(
    valid_project: Path,
) -> None:
    snapshot = inspect_project(valid_project)
    rendered = render_project_report(snapshot)

    assert "No findings were reported by the checks that ran." in rendered


def test_render_project_report_counts_unknown_resources(
    valid_project: Path,
) -> None:
    notes_path = valid_project / "videos" / "notes.txt"
    notes_path.write_text(
        "notes",
        encoding="utf-8",
    )

    snapshot = inspect_project(valid_project)
    rendered = render_project_report(snapshot)

    assert "unknown_file: 1" in rendered
    assert "? notes.txt [unknown_file]" in rendered


def test_render_project_report_can_hide_unknown_resources_from_tree(
    valid_project: Path,
) -> None:
    notes_path = valid_project / "videos" / "notes.txt"
    notes_path.write_text(
        "notes",
        encoding="utf-8",
    )

    snapshot = inspect_project(valid_project)
    rendered = render_project_report(
        snapshot,
        show_unknown=False,
    )

    # Unknown resources remain counted in the report summary.
    assert "unknown_file: 1" in rendered

    # The individual unknown entry is hidden
    assert "? notes.txt [unknown_file]" not in rendered


def test_render_project_report_can_show_schema_ids(
    valid_project: Path,
) -> None:
    snapshot = inspect_project(valid_project)
    rendered = render_project_report(
        snapshot,
        show_schema_ids=True,
    )

    assert "dlc.project" in rendered
    assert "dlc.project.config" in rendered


def test_render_project_report_limits_repetitive_children(
    valid_project: Path,
) -> None:
    videos = valid_project / "videos"

    for index in range(4):
        (videos / f"session-{index:02d}.mp4").touch()

    snapshot = inspect_project(valid_project)
    rendered = render_project_report(
        snapshot,
        max_children_per_kind=2,
    )

    assert "session-00.mp4" in rendered
    assert "session-01.mp4" in rendered
    assert "session-02.mp4" not in rendered
    assert "2 additional children omitted" in rendered

    # Tree truncation must not affect the resource summary.
    assert "source_video: 4" in rendered


def test_default_report_hides_known_unmanaged_metadata(
    valid_project: Path,
) -> None:
    dataset = valid_project / "labeled-data" / "session-01"
    dataset.mkdir()

    (dataset / ".napari-deeplabcut.json").write_text(
        "{}",
        encoding="utf-8",
    )

    rendered = render_project_report(inspect_project(valid_project))

    assert ".napari-deeplabcut.json" not in rendered
    assert "napari_dlc_metadata" not in rendered


def test_detailed_report_shows_known_unmanaged_metadata(
    valid_project: Path,
) -> None:
    dataset = valid_project / "labeled-data" / "session-01"
    dataset.mkdir()

    (dataset / ".napari-deeplabcut.json").write_text(
        "{}",
        encoding="utf-8",
    )

    rendered = render_project_report(
        inspect_project(valid_project),
        show_unmanaged=True,
    )

    assert ".napari-deeplabcut.json" in rendered
    assert "napari_dlc_metadata: 1" in rendered
    assert "known_unmanaged" in rendered


@pytest.mark.xfail(
    reason="the report carries no provenance stamp",
    strict=True,
    raises=AssertionError,
)
def test_render_project_report_stamps_the_snapshot_version(
    valid_project: Path,
) -> None:
    snapshot = inspect_project(valid_project)
    rendered = render_project_report(snapshot)

    assert "snapshot version" in rendered.lower()
    assert snapshot.version in rendered


@pytest.mark.xfail(
    reason="the report prints a finding's instance the same way whatever its reach, so every frame of ds1 (ALL_OF) "
    "reads exactly like one finding about ds1 (ONE_OF)",
    strict=True,
    raises=AssertionError,
)
def test_render_project_report_tells_all_of_apart_from_one_of(
    tmp_path: Path,
) -> None:
    code = StructureFindingCode.PROJECT_NODE_UNREADABLE
    snapshot = inspect_project(tmp_path)

    def rendered_line(reach: Reach) -> str:
        finding = Finding(
            code=code,
            schema_id="dlc.labeled_data.frame",
            reach=reach,
            instance=("labeled-data", "ds1"),
            evidence=dict.fromkeys(code.evidence_keys, "placeholder"),
        )
        report = render_project_report(snapshot.model_copy(update={"findings": (finding,)}))

        return next(line for line in report.splitlines() if "PROJECT_NODE_UNREADABLE" in line)

    assert rendered_line(Reach.ALL_OF) != rendered_line(Reach.ONE_OF)
