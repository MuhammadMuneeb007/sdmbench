"""The R bridge: run R scripts and exchange tabular data with them."""

from sdmbench.rbridge.runner import SCRIPTS_DIR, RRunner, RScriptResult, find_rscript

__all__ = ["RRunner", "RScriptResult", "SCRIPTS_DIR", "find_rscript"]
