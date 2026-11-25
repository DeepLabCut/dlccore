"""Schema-driven, read-only DeepLabCut project inspection."""

from dlc_octopus.api import inspect_project
from dlc_octopus.capabilities import get_project_manager_capabilities
from dlc_octopus.schemas.snapshots import ProjectSnapshot

__all__ = ["ProjectSnapshot", "get_project_manager_capabilities", "inspect_project"]
