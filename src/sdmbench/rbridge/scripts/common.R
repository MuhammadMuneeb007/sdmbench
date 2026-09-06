# common.R -- shared helpers for every sdmbench R bridge script.
#
# Contract with the Python side (see sdmbench/rbridge/runner.py):
#   Rscript <script>.R <input.json> <output.json>
# The script reads its parameters from input.json and writes its result to
# output.json. A non-empty `error` field in output.json marks failure; the
# Python side turns that into a SKIPPED/FAILED result row rather than crashing
# the whole benchmark.
#
# These helpers deliberately avoid any dependency beyond `jsonlite`, so that a
# missing modelling package produces a clean structured error instead of an
# R-level parse failure.

sdmbench_args <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  if (length(args) < 2L) {
    stop("usage: Rscript <script>.R <input.json> <output.json>", call. = FALSE)
  }
  list(input = args[[1L]], output = args[[2L]])
}

sdmbench_require <- function(pkgs) {
  # Check availability WITHOUT installing. sdmbench never installs R packages.
  missing <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly = TRUE)]
  if (length(missing) > 0L) {
    stop(
      sprintf(
        "missing R package(s): %s. Install them with: Rscript scripts/setup_r_packages.R",
        paste(missing, collapse = ", ")
      ),
      call. = FALSE
    )
  }
  invisible(TRUE)
}

sdmbench_read_input <- function(path) {
  if (!requireNamespace("jsonlite", quietly = TRUE)) {
    stop("the R package 'jsonlite' is required by the sdmbench R bridge", call. = FALSE)
  }
  jsonlite::fromJSON(path, simplifyVector = TRUE)
}

sdmbench_write_output <- function(obj, path) {
  txt <- jsonlite::toJSON(obj, auto_unbox = TRUE, digits = NA, na = "null", null = "null")
  writeLines(txt, path)
  invisible(path)
}

sdmbench_write_error <- function(msg, path) {
  # Always emit a parseable output.json, even on failure -- the Python side
  # reads the message out of it to populate `status_detail`.
  if (requireNamespace("jsonlite", quietly = TRUE)) {
    writeLines(
      jsonlite::toJSON(list(error = as.character(msg)), auto_unbox = TRUE),
      path
    )
  } else {
    writeLines(sprintf('{"error": "%s"}', gsub('"', "'", as.character(msg))), path)
  }
  invisible(path)
}

# Wrap a script body so that any condition becomes a structured error.
sdmbench_main <- function(fn) {
  args <- sdmbench_args()
  result <- tryCatch(
    {
      input <- sdmbench_read_input(args$input)
      fn(input)
    },
    error = function(e) structure(list(error = conditionMessage(e)), class = "sdmbench_error"),
    warning = function(w) {
      # Warnings are informative, not fatal: re-run the body with warnings muffled.
      withCallingHandlers(
        {
          input <- sdmbench_read_input(args$input)
          fn(input)
        },
        warning = function(w2) invokeRestart("muffleWarning")
      )
    }
  )
  if (inherits(result, "sdmbench_error")) {
    sdmbench_write_error(result$error, args$output)
    quit(status = 1L, save = "no")
  }
  sdmbench_write_output(result, args$output)
  invisible(NULL)
}

# --- data exchange -----------------------------------------------------------

sdmbench_read_frame <- function(path, categorical = character(0)) {
  df <- utils::read.csv(path, stringsAsFactors = FALSE, na.strings = c("NA", ""))
  for (nm in intersect(categorical, names(df))) {
    df[[nm]] <- as.factor(df[[nm]])
  }
  df
}

# Align factor levels of the evaluation data to the training data.
#
# This mirrors the approach in the disdat vignette: evaluation data may not
# contain every level seen in training, and a model must still be able to
# predict for it. Levels are taken from training only -- never the reverse,
# which would leak test information into the fitted representation.
sdmbench_align_factors <- function(train, test, categorical) {
  for (nm in intersect(categorical, intersect(names(train), names(test)))) {
    lv <- levels(as.factor(train[[nm]]))
    train[[nm]] <- factor(train[[nm]], levels = lv)
    test[[nm]] <- factor(test[[nm]], levels = lv)
  }
  list(train = train, test = test)
}

# Presence/background weighting used by BRT and GAM in Dinnage & Warren (2026),
# following Valavi et al. (2022): background points are down-weighted by the
# ratio of presences to backgrounds so the two classes carry equal total mass.
sdmbench_pb_weights <- function(y) {
  y <- as.integer(y)
  n_pres <- sum(y == 1L)
  n_bg <- sum(y == 0L)
  if (n_pres == 0L || n_bg == 0L) {
    return(rep(1, length(y)))
  }
  ifelse(y == 1L, 1, n_pres / n_bg)
}
