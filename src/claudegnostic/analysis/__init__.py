"""Pure-function analysis layer over the claudegnostic DuckDB schema.

Surfaces (report, dashboard) import from here. Modules never render, never
write, and never open connections they do not close. Every public function
returns a polars DataFrame with a fixed column set, including the empty case.
"""
