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

from __future__ import annotations

from dlccore.project.schemas.base.observations import (
    NodeStatus,
    ObservedNode,
)
from dlccore.project.schemas.base.resources import ProjectResource


def flatten_observed_project(
    project: ObservedNode,
    *,
    include_project_root: bool = False,
) -> tuple[ProjectResource, ...]:
    """Return a flat projection of physical project resources.

    Any schema node with an actual filesystem entry is included -
    present, wrong-type, or unreadable alike - along with unclaimed
    immediate children. Only nodes with no filesystem entry at all
    (``NodeStatus.MISSING``) are retained solely in the semantic
    observation tree.

    ``ProjectResource.kind`` reports the schema-expected kind even when
    the underlying entry does not actually match it; validity is the
    job of ``project`` and its findings, not of this projection.

    Unknown directories are included without recursively interpreting
    their contents.
    """

    resources: list[ProjectResource] = []

    if include_project_root:
        _collect_resources(
            node=project,
            resources=resources,
        )
    else:
        resources.extend(project.unclaimed_resources)

        for child in project.schema_children:
            _collect_resources(
                node=child,
                resources=resources,
            )

    return tuple(
        sorted(
            resources,
            key=lambda resource: (
                resource.project_relative_path.as_posix().casefold(),
                resource.project_relative_path.as_posix(),
            ),
        )
    )


def _collect_resources(
    *,
    node: ObservedNode,
    resources: list[ProjectResource],
) -> None:
    """Recursively collect physical resources from one observed node."""

    is_physical_resource = node.status != NodeStatus.MISSING

    if is_physical_resource:
        resources.append(
            ProjectResource(
                project_relative_path=(node.project_relative_path),
                kind=node.kind,
                management=node.management,
                symlink=node.symlink,
            )
        )

    resources.extend(node.unclaimed_resources)

    for child in node.schema_children:
        _collect_resources(
            node=child,
            resources=resources,
        )
