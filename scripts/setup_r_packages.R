# setup_r_packages.R -- install the R packages sdmbench's R bridge uses.
#
#   Rscript scripts/setup_r_packages.R              # required + recommended
#   Rscript scripts/setup_r_packages.R --check      # report only, install nothing
#   Rscript scripts/setup_r_packages.R --all        # include optional extras
#
# This script is the ONLY place sdmbench installs anything into R, and it only
# runs when you invoke it. The Python side never installs R packages: it
# detects what is missing and reports it, so a missing package becomes a
# SKIPPED_DEPENDENCY result row rather than a surprise mutation of your library.
#
# Why these packages
# ------------------
# Dinnage & Warren (2026) sec. 2.2.1 followed "the best-practice configurations
# specified in Valavi et al. (2022)". Those configurations are properties of
# specific R packages -- maxnet's elastic-net MaxEnt, dismo::gbm.step's
# CV-selected tree count, randomForest's per-class down-sampling, mgcv's
# penalised thin-plate splines with REML. Reimplementing them in Python would
# change the published numbers, so strict reproduction calls the real thing.

args <- commandArgs(trailingOnly = TRUE)
check_only <- "--check" %in% args
install_all <- "--all" %in% args

REQUIRED <- c(
  jsonlite     = "the R bridge protocol (JSON exchange with Python)",
  disdat       = "the benchmark dataset: 226 species, 6 regions (Elith et al. 2020)"
)

PUBLISHED_MODELS <- c(
  maxnet       = "MaxEnt via elastic-net GLM (paper sec. 2.2.1)",
  dismo        = "gbm.step() for BRT, and the Java MaxEnt interface",
  gbm          = "the boosting engine dismo::gbm.step builds on",
  randomForest = "random forest with per-class down-sampling",
  mgcv         = "GAMs with penalised thin-plate regression splines"
)

SPATIAL <- c(
  sf           = "spatial features; the reference 10 km buffer implementation",
  terra        = "raster handling"
)

VALIDATION <- c(
  yardstick    = "the metric implementations the paper used (ROC-AUC, PR-AUC)",
  tidysdm      = "boyce_cont(): reference Continuous Boyce Index for parity tests"
)

OPTIONAL <- c(
  blockCV      = "spatial block cross-validation",
  spatialsample = "spatial resampling (paper sec. 2.7)",
  rsample      = "stratified holdout splits (paper sec. 2.7)",
  ENMeval      = "MaxEnt tuning and evaluation",
  biomod2      = "ensemble SDM platform",
  rJava        = "required by dismo::maxent() for the original Java MaxEnt",
  reticulate   = "calling Python from R (the authors' own pipeline used this)"
)

groups <- list(
  "REQUIRED" = REQUIRED,
  "PUBLISHED BASELINES" = PUBLISHED_MODELS,
  "SPATIAL" = SPATIAL,
  "VALIDATION / METRIC PARITY" = VALIDATION
)
if (install_all) {
  groups[["OPTIONAL"]] <- OPTIONAL
}

installed_version <- function(pkg) {
  tryCatch(as.character(utils::packageVersion(pkg)), error = function(e) NA_character_)
}

cat("sdmbench R dependency check\n")
cat("R version:", paste(R.version$major, R.version$minor, sep = "."), "\n")
cat("Library paths:\n")
for (p in .libPaths()) cat("  ", p, "\n")
cat("\n")

missing <- character(0)
for (group_name in names(groups)) {
  cat(group_name, "\n")
  pkgs <- groups[[group_name]]
  for (pkg in names(pkgs)) {
    version <- installed_version(pkg)
    status <- if (is.na(version)) "MISSING" else version
    cat(sprintf("  %-14s %-12s %s\n", pkg, status, pkgs[[pkg]]))
    if (is.na(version)) missing <- c(missing, pkg)
  }
  cat("\n")
}

if (!install_all) {
  cat("OPTIONAL (not checked; pass --all to include)\n")
  for (pkg in names(OPTIONAL)) {
    cat(sprintf("  %-14s %s\n", pkg, OPTIONAL[[pkg]]))
  }
  cat("\n")
}

if (length(missing) == 0L) {
  cat("All checked packages are installed. Nothing to do.\n")
  quit(status = 0L, save = "no")
}

cat(sprintf("%d package(s) missing: %s\n\n", length(missing),
            paste(missing, collapse = ", ")))

if (check_only) {
  cat("--check given: nothing was installed.\n")
  cat("Re-run without --check to install.\n")
  quit(status = 1L, save = "no")
}

repo <- getOption("repos")[["CRAN"]]
if (is.null(repo) || is.na(repo) || repo == "@CRAN@") {
  repo <- "https://cloud.r-project.org"
}
cat("Installing from:", repo, "\n")
cat("Target library:", .libPaths()[1L], "\n\n")

for (pkg in missing) {
  cat("Installing", pkg, "...\n")
  # Keep going on failure: sf and terra need system libraries (GDAL, GEOS,
  # PROJ) that a plain install.packages() cannot provide, and one failure
  # should not stop the rest.
  ok <- tryCatch(
    {
      utils::install.packages(pkg, repos = repo, quiet = FALSE)
      !is.na(installed_version(pkg))
    },
    error = function(e) {
      cat("  FAILED:", conditionMessage(e), "\n")
      FALSE
    },
    warning = function(w) {
      cat("  WARNING:", conditionMessage(w), "\n")
      !is.na(installed_version(pkg))
    }
  )
  cat(if (ok) "  ok\n" else "  not installed\n")
}

cat("\nFinal state:\n")
still_missing <- character(0)
for (pkg in missing) {
  version <- installed_version(pkg)
  cat(sprintf("  %-14s %s\n", pkg, if (is.na(version)) "STILL MISSING" else version))
  if (is.na(version)) still_missing <- c(still_missing, pkg)
}

if (length(still_missing) > 0L) {
  cat("\nSome packages could not be installed.\n")
  if (any(c("sf", "terra") %in% still_missing)) {
    cat("  sf and terra need system libraries. The reliable route is conda:\n")
    cat("      conda install -c conda-forge r-sf r-terra\n")
  }
  if ("rJava" %in% still_missing) {
    cat("  rJava needs a JDK and `R CMD javareconf`.\n")
  }
  cat("  sdmbench will report the affected models as SKIPPED_DEPENDENCY\n")
  cat("  and continue with everything else.\n")
  quit(status = 1L, save = "no")
}

cat("\nDone. Verify from Python with:  sdmbench env check\n")
