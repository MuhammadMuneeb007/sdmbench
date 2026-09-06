# randomforest.R -- Random Forest with class-balanced down-sampling.
#
# Settings follow Dinnage & Warren (2026) sec. 2.2.1 verbatim:
#
#   "Random Forest models used the randomForest package with balanced sampling.
#    Rather than weighting, we employed per-class downsampling: for each of
#    1,000 trees, both presence and absence classes were sampled to the size of
#    the minority class. This approach, which directly addresses class
#    imbalance at the tree level, used default parameters for variable
#    selection and node size."
#
# This is the "RF down-sampled" configuration of Valavi et al. (2022), which is
# a different model from a plain random forest -- do not substitute
# sklearn.ensemble.RandomForestClassifier for it in strict reproduction mode.
#
# NOTE ON `replace`: the paper specifies the sample size but not whether
# sampling is with replacement. randomForest's default is replace = TRUE, and
# that default is used here; sdmbench flags it as UNVERIFIED in the provenance
# table.

.script_dir <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg) > 0L) dirname(sub("^--file=", "", file_arg[[1L]])) else getwd()
}
source(file.path(.script_dir(), "common.R"))

`%||%` <- function(a, b) if (is.null(a)) b else a

main <- function(input) {
  sdmbench_require(c("randomForest", "jsonlite"))

  categorical <- if (is.null(input$categorical)) character(0) else as.character(input$categorical)
  predictors <- as.character(input$predictors)

  train <- sdmbench_read_frame(input$inputs$train, categorical)
  test <- sdmbench_read_frame(input$inputs$test, categorical)
  aligned <- sdmbench_align_factors(train, test, categorical)
  train <- aligned$train
  test <- aligned$test

  y <- factor(as.integer(train$occ), levels = c(0L, 1L))
  x <- train[, predictors, drop = FALSE]

  n_pres <- sum(y == "1")
  n_bg <- sum(y == "0")
  minority <- min(n_pres, n_bg)
  if (minority < 1L) stop("training data is single-class; cannot fit a classifier")

  ntree <- as.integer(input$ntree %||% 1000L)
  replace <- if (is.null(input$replace)) TRUE else isTRUE(input$replace)
  # Named by class label so randomForest maps sizes to the right strata.
  sampsize <- c("0" = minority, "1" = minority)

  set.seed(as.integer(input$seed %||% 32639L))

  fit_start <- Sys.time()
  model <- randomForest::randomForest(
    x = x, y = y,
    ntree = ntree,
    sampsize = sampsize,
    replace = replace
  )
  fit_seconds <- as.numeric(difftime(Sys.time(), fit_start, units = "secs"))

  pred_start <- Sys.time()
  probs <- stats::predict(model, test[, predictors, drop = FALSE], type = "prob")
  preds <- as.numeric(probs[, "1"])
  predict_seconds <- as.numeric(difftime(Sys.time(), pred_start, units = "secs"))

  list(
    predictions = preds,
    fit_seconds = fit_seconds,
    predict_seconds = predict_seconds,
    model_version = as.character(utils::packageVersion("randomForest")),
    hyperparameters = list(
      ntree = ntree,
      sampsize = as.integer(minority),
      replace = replace,
      strategy = "per_class_downsampling",
      mtry = as.integer(model$mtry)
    )
  )
}

sdmbench_main(main)
