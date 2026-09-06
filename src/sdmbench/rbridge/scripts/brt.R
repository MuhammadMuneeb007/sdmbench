# brt.R -- Boosted Regression Trees via dismo::gbm.step().
#
# Settings follow Dinnage & Warren (2026) sec. 2.2.1 verbatim:
#
#   "implemented using gbm.step() from the dismo package, following Valavi et
#    al. (2022). To account for class imbalance, we down-weighted background
#    points relative to presence points by the ratio of presences to
#    backgrounds. Tree complexity was set adaptively: stumps (complexity = 1)
#    for species with fewer than 50 presences, moderate interactions
#    (complexity = 5) for larger samples. We used a learning rate of 0.001,
#    stochastic gradient boosting with 75% bagging fraction, and 5-fold
#    cross-validation to select the optimal number of trees up to a maximum of
#    10,000."
#
# gbm.step() performs its own internal cross-validation on the TRAINING data
# only to choose the number of trees. That is legitimate model selection; the
# independent presence-absence survey data never enters it.

.script_dir <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg) > 0L) dirname(sub("^--file=", "", file_arg[[1L]])) else getwd()
}
source(file.path(.script_dir(), "common.R"))

`%||%` <- function(a, b) if (is.null(a)) b else a

main <- function(input) {
  sdmbench_require(c("dismo", "gbm", "jsonlite"))

  categorical <- if (is.null(input$categorical)) character(0) else as.character(input$categorical)
  predictors <- as.character(input$predictors)

  train <- sdmbench_read_frame(input$inputs$train, categorical)
  test <- sdmbench_read_frame(input$inputs$test, categorical)
  aligned <- sdmbench_align_factors(train, test, categorical)
  train <- aligned$train
  test <- aligned$test

  train$occ <- as.integer(train$occ)
  n_pres <- sum(train$occ == 1L)

  # Adaptive tree complexity: stumps below 50 presences, depth 5 above.
  tree_complexity <- if (!is.null(input$tree_complexity)) {
    as.integer(input$tree_complexity)
  } else if (n_pres < 50L) 1L else 5L

  learning_rate <- as.numeric(input$learning_rate %||% 0.001)
  bag_fraction <- as.numeric(input$bag_fraction %||% 0.75)
  n_folds <- as.integer(input$n_folds %||% 5L)
  max_trees <- as.integer(input$max_trees %||% 10000L)

  weights <- sdmbench_pb_weights(train$occ)

  set.seed(as.integer(input$seed %||% 32639L))
  model_data <- train[, c("occ", predictors), drop = FALSE]

  fit_start <- Sys.time()
  model <- dismo::gbm.step(
    data = model_data,
    gbm.x = predictors,
    gbm.y = "occ",
    family = "bernoulli",
    tree.complexity = tree_complexity,
    learning.rate = learning_rate,
    bag.fraction = bag_fraction,
    n.folds = n_folds,
    max.trees = max_trees,
    site.weights = weights,
    silent = TRUE,
    plot.main = FALSE
  )
  fit_seconds <- as.numeric(difftime(Sys.time(), fit_start, units = "secs"))

  if (is.null(model)) {
    # gbm.step returns NULL when it cannot find a usable number of trees at the
    # requested learning rate -- a real, reportable outcome, not a crash.
    stop("gbm.step() failed to converge (returned NULL); try a larger learning rate")
  }

  pred_start <- Sys.time()
  preds <- as.numeric(gbm::predict.gbm(
    model, test[, predictors, drop = FALSE],
    n.trees = model$gbm.call$best.trees, type = "response"
  ))
  predict_seconds <- as.numeric(difftime(Sys.time(), pred_start, units = "secs"))

  list(
    predictions = preds,
    fit_seconds = fit_seconds,
    predict_seconds = predict_seconds,
    model_version = as.character(utils::packageVersion("dismo")),
    hyperparameters = list(
      tree_complexity = tree_complexity,
      learning_rate = learning_rate,
      bag_fraction = bag_fraction,
      n_folds = n_folds,
      max_trees = max_trees,
      best_trees = model$gbm.call$best.trees,
      weighting = "presence_background_ratio"
    )
  )
}

sdmbench_main(main)
