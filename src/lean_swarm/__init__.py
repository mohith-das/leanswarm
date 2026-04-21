"""Backward-compatible alias for the renamed `leanswarm` package."""

from __future__ import annotations

import sys
from warnings import warn

import leanswarm as _leanswarm

warn(
    "`lean_swarm` is deprecated; import `leanswarm` instead.",
    DeprecationWarning,
    stacklevel=2,
)

sys.modules[__name__] = _leanswarm
