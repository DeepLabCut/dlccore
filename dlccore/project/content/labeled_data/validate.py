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

"""One labeled-data file's findings: the file reports, then the model reports when the file has a model."""

from __future__ import annotations

from dlccore.io.labeled_data import LabeledDataFile
from dlccore.project.content.labeled_data import file, model
from dlccore.project.schemas.base.findings import Finding
from dlccore.project.schemas.locations import Location


def validate_labeled_data_file(data: LabeledDataFile, at: Location) -> tuple[Finding, ...]:
    """Return every finding about one file that was read, anchored at ``at``.

    A file with no model gets only its file findings: those say why it has none.
    """

    findings = [finding for report in file.REPORTS for finding in report.emit(data, at)]
    try:
        labels = data.labeled_data
    except ValueError:
        return tuple(findings)
    findings += [finding for report in model.REPORTS for finding in report.emit(data, labels, at)]
    return tuple(findings)
