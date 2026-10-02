"""quantmet: quantitative meteorology tools extracted from the real-time
diagnostics pipelines behind scorvec.com.

Modules are imported individually (``from quantmet.tem import tem_terms``);
importing the package itself pulls in nothing beyond the version, so the
optional dependencies of one module (pyshtools, xarray for helmholtz and the
streamfunction path of waf) never burden another.
"""
__version__ = "0.2.0"
