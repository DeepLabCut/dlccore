"""What a consumer composes with the entry points in ``dlc_octopus.api``: scopes and its own finding policy."""

from dlc_octopus.policy import get_blocking_findings, unhandled
from dlc_octopus.schemas.scope import Scope

__all__ = ["Scope", "get_blocking_findings", "unhandled"]
