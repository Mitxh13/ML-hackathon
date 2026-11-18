"""Lane A data utilities, imported lazily to keep core loaders lightweight."""
from __future__ import annotations

_EXPORTS = {
    "load_track": ("loaders", "load_track"), "load_nodes": ("loaders", "load_nodes"),
    "load_arcs": ("loaders", "load_arcs"), "load_route": ("loaders", "load_route"),
    "load_trip": ("loaders", "load_trip"),
    "project_coordinates": ("preprocessing", "project_coordinates"),
    "preprocess_track": ("preprocessing", "preprocess_track"),
    "assign_point_ground_truth": ("ground_truth", "assign_point_ground_truth"),
    "CandidateGenerator": ("candidates", "CandidateGenerator"),
    "extract_candidate_features": ("features", "extract_candidate_features"),
    "plot_missing_values": ("eda_plots", "plot_missing_values"),
    "plot_speed_distribution": ("eda_plots", "plot_speed_distribution"),
    "plot_candidate_distances": ("eda_plots", "plot_candidate_distances"),
}
__all__ = list(_EXPORTS)


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module
    module, attr = _EXPORTS[name]
    value = getattr(import_module(f"{__name__}.{module}"), attr)
    globals()[name] = value
    return value
