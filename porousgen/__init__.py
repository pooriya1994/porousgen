"""PorousGen — LBM-ready porous-media geometry generator."""
from .generator import (
    __version__, __author__, __citation__, GENERATOR_INFO,
    generate_granular, generate_fibrous, generate_cellular,
    generate_consolidated, generate_ordered_gyroid,
    generate_open_foam, generate_blob, generate_overlapping_spheres,
    export_palabos, read_palabos, export_stl, export_all, write_info_file,
    compute_porosity, visualize_catalog, visualize_3d_voxels,
)
from .metrics import compute_metrics, format_metrics, local_thickness

__all__ = [
    "__version__", "__author__", "__citation__", "GENERATOR_INFO",
    "generate_granular", "generate_fibrous", "generate_cellular",
    "generate_consolidated", "generate_ordered_gyroid",
    "generate_open_foam", "generate_blob", "generate_overlapping_spheres",
    "export_palabos", "read_palabos", "export_stl", "export_all", "write_info_file",
    "compute_porosity", "visualize_catalog", "visualize_3d_voxels",
    "compute_metrics", "format_metrics", "local_thickness",
]
