# spatial_split.R -- reference implementation of the 10 km spatial filter in R.
#
# Dinnage & Warren (2026) sec. 2.1.2 / 2.5:
#
#   "we created spatially-filtered training sets by excluding training points
#    within 10 km of any test location"
#
# The authors built their spatial features with `sf`. sdmbench implements the
# same filter natively in Python (see sdmbench/splits/spatial.py) so that the
# core benchmark does not require R, but the two must agree -- distance on a
# geographic CRS is computed differently by different libraries, and a
# disagreement here silently changes which species survive the filter.
#
# This script is the R reference for that parity test
# (tests/test_spatial_split_parity_r.py, marked `@pytest.mark.r`).

.script_dir <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg) > 0L) dirname(sub("^--file=", "", file_arg[[1L]])) else getwd()
}
source(file.path(.script_dir(), "common.R"))

`%||%` <- function(a, b) if (is.null(a)) b else a

main <- function(input) {
  sdmbench_require(c("sf", "jsonlite"))

  train <- utils::read.csv(input$inputs$train, stringsAsFactors = FALSE)
  test <- utils::read.csv(input$inputs$test, stringsAsFactors = FALSE)
  crs <- input$crs
  buffer_m <- as.numeric(input$buffer_m %||% 10000)

  train_sf <- sf::st_as_sf(train, coords = c("x", "y"), crs = crs)
  test_sf <- sf::st_as_sf(test, coords = c("x", "y"), crs = crs)

  # st_is_within_distance uses Euclidean distance for projected CRSs and
  # great-circle (s2) distance for geographic CRSs -- which is exactly the
  # behaviour sdmbench's Python implementation reproduces.
  near <- sf::st_is_within_distance(train_sf, test_sf, dist = buffer_m)
  excluded <- lengths(near) > 0L

  list(
    n_train = nrow(train),
    n_test = nrow(test),
    buffer_m = buffer_m,
    crs = crs,
    is_geographic = isTRUE(sf::st_is_longlat(train_sf)),
    n_excluded = sum(excluded),
    n_retained = sum(!excluded),
    # 1-based row indices retained, for direct comparison against Python.
    retained_indices = which(!excluded),
    sf_version = as.character(utils::packageVersion("sf"))
  )
}

sdmbench_main(main)
