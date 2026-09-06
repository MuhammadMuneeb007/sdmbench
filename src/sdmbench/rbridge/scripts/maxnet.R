# maxnet.R -- MaxEnt via the `maxnet` R package.
#
# Settings follow Dinnage & Warren (2026) sec. 2.2.1 verbatim:
#
#   "We used the default regularization multiplier (regmult = 1) and included
#    all default feature classes: linear, quadratic, product, threshold, and
#    hinge. Predictions used the complementary log-log link function."
#
# maxnet's default `classes` string is "lqpht" (linear, quadratic, product,
# hinge, threshold), which is exactly that set. We pass it explicitly rather
# than relying on the default so the setting is visible in the results.

.script_dir <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg) > 0L) dirname(sub("^--file=", "", file_arg[[1L]])) else getwd()
}
source(file.path(.script_dir(), "common.R"))

main <- function(input) {
  sdmbench_require(c("maxnet", "jsonlite"))

  categorical <- if (is.null(input$categorical)) character(0) else as.character(input$categorical)
  predictors <- as.character(input$predictors)

  train <- sdmbench_read_frame(input$inputs$train, categorical)
  test <- sdmbench_read_frame(input$inputs$test, categorical)
  aligned <- sdmbench_align_factors(train, test, categorical)
  train <- aligned$train
  test <- aligned$test

  y <- as.integer(train$occ)
  x <- train[, predictors, drop = FALSE]

  regmult <- if (is.null(input$regmult)) 1 else as.numeric(input$regmult)
  classes <- if (is.null(input$classes)) "lqpht" else as.character(input$classes)

  set.seed(as.integer(input$seed %||% 32639L))

  fit_start <- Sys.time()
  model <- maxnet::maxnet(p = y, data = x, regmult = regmult,
                          f = maxnet::maxnet.formula(y, x, classes = classes))
  fit_seconds <- as.numeric(difftime(Sys.time(), fit_start, units = "secs"))

  pred_start <- Sys.time()
  # `cloglog` is the output transform used in the paper.
  preds <- as.numeric(stats::predict(model, test[, predictors, drop = FALSE],
                                     type = "cloglog", clamp = FALSE))
  predict_seconds <- as.numeric(difftime(Sys.time(), pred_start, units = "secs"))

  list(
    predictions = preds,
    fit_seconds = fit_seconds,
    predict_seconds = predict_seconds,
    model_version = as.character(utils::packageVersion("maxnet")),
    hyperparameters = list(regmult = regmult, classes = classes, link = "cloglog")
  )
}

`%||%` <- function(a, b) if (is.null(a)) b else a

sdmbench_main(main)
