# maxent_java.R -- the original Java MaxEnt via dismo::maxent().
#
# Dinnage & Warren (2026) sec. 2.2.1:
#
#   "MaxEnt (Java) is the original Java-based MaxEnt implementation accessed via
#    the dismo package. This differs from MaxNet in its optimization approach
#    and handling of regularization. We used automatic feature selection and
#    cloglog output format for comparability."
#
# dismo::maxent() requires maxent.jar to be installed by hand into
#   system.file("java", package = "dismo")
# because its licence does not permit redistribution. This script checks for it
# and fails with instructions rather than a confusing Java stack trace.

.script_dir <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg) > 0L) dirname(sub("^--file=", "", file_arg[[1L]])) else getwd()
}
source(file.path(.script_dir(), "common.R"))

`%||%` <- function(a, b) if (is.null(a)) b else a

main <- function(input) {
  sdmbench_require(c("dismo", "rJava", "jsonlite"))

  jar <- file.path(system.file("java", package = "dismo"), "maxent.jar")
  if (!file.exists(jar)) {
    stop(
      paste0(
        "maxent.jar not found at ", jar, ".\n",
        "  Download it from https://biodiversityinformatics.amnh.org/open_source/maxent/ ",
        "and copy maxent.jar into that directory. Its licence does not permit ",
        "redistribution, so sdmbench cannot install it for you."
      ),
      call. = FALSE
    )
  }

  categorical <- if (is.null(input$categorical)) character(0) else as.character(input$categorical)
  predictors <- as.character(input$predictors)

  train <- sdmbench_read_frame(input$inputs$train, categorical)
  test <- sdmbench_read_frame(input$inputs$test, categorical)
  aligned <- sdmbench_align_factors(train, test, categorical)
  train <- aligned$train
  test <- aligned$test

  y <- as.integer(train$occ)
  x <- train[, predictors, drop = FALSE]

  # "cloglog" output and automatic feature selection (the dismo default).
  extra_args <- input$args %||% c("outputformat=cloglog")

  set.seed(as.integer(input$seed %||% 32639L))

  fit_start <- Sys.time()
  model <- dismo::maxent(x = x, p = y, args = as.character(extra_args))
  fit_seconds <- as.numeric(difftime(Sys.time(), fit_start, units = "secs"))

  pred_start <- Sys.time()
  preds <- as.numeric(dismo::predict(model, test[, predictors, drop = FALSE]))
  predict_seconds <- as.numeric(difftime(Sys.time(), pred_start, units = "secs"))

  list(
    predictions = preds,
    fit_seconds = fit_seconds,
    predict_seconds = predict_seconds,
    model_version = as.character(utils::packageVersion("dismo")),
    hyperparameters = list(
      args = as.character(extra_args),
      feature_selection = "automatic",
      output = "cloglog"
    )
  )
}

sdmbench_main(main)
