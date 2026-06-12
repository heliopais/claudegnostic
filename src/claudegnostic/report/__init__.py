"""HTML report surface.

Renders a single self-contained ``.html`` file summarising the user's Claude
Code usage across three lenses (Cost, Productivity, Workflow Archaeology).
All images are base64-embedded PNGs; no sidecar directories, no network.
"""

from claudegnostic.report.render import render_report

__all__ = ["render_report"]
