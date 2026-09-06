# gam.R -- Generalised Additive Model via mgcv.
#
# Settings follow Dinnage & Warren (2026) sec. 2.2.1:
#
#   "fitted using the mgcv package and using the same region-specific R model
#    formulas used in Valavi et al. (2022), which specified smooth terms for
#    continuous predictors and linear factor terms for categorical variables.
#    Continuous predictors were modeled using penalized thin plate regression
#    splines, while categorical variables were included as factor terms. Models
#    used a binomial family with logit link, restricted maximum likelihood for
#    smoothing parameter estimation, and the same presence-background weighting
#    as BRT."
#
# PROVENANCE WARNING
# ------------------
# The paper defers the exact per-region formulas to Valavi et al. (2022) and the
# authors' repository, which was not public when this script was written. The
# formula constructed below implements the *description* in the paper --
# s(<continuous>, bs = "tp") + <categorical as factor> -- but the basis
# dimension `k` per predictor could not be verified. Pass `formula` in the input
# payload to override with the authoritative formula once available; sdmbench
# records this as an UNVERIFIED constant in the provenance table either way.

.script_dir <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg) > 0L) dirname(sub("^--file=", "", file_arg[[1L]])) else getwd()
}
source(file.path(.script_dir(), "common.R"))

`%||%` <- function(a, b) if (is.null(a)) b else a

build_formula <- function(predictors, categorical, train, k = NULL) {
  terms <- character(0)
  for (p in predictors) {
    if (p %in% categorical) {
      # Categorical predictors enter as linear factor terms. A factor with a
      # single observed level carries no information and would break mgcv.
      if (nlevels(droplevels(as.factor(train[[p]]))) > 1L) {
        terms <- c(terms, p)
      }
    } else {
      n_unique <- length(unique(stats::na.omit(train[[p]])))
      if (n_unique < 3L) {
        # Too few distinct values to smooth: enter linearly.
        if (n_unique > 1L) terms <- c(terms, p)
      } else {
        # Cap the basis dimension at the number of distinct values, as mgcv
        # requires k <= number of unique covariate values.
        kk <- if (is.null(k)) min(10L, n_unique - 1L) else min(as.integer(k), n_unique - 1L)
        terms <- c(terms, sprintf('s(%s, bs = "tp", k = %d)', p, kk))
      }
    }
  }
  if (length(terms) == 0L) stop("no usable predictors for GAM after screening")
  stats::as.formula(paste("occ ~", paste(terms, collapse = " + ")))
}

main <- function(input) {
  sdmbench_require(c("mgcv", "jsonlite"))

  categorical <- if (is.null(input$categorical)) character(0) else as.character(input$categorical)
  predictors <- as.character(input$predictors)

  train <- sdmbench_read_frame(input$inputs$train, categorical)
  test <- sdmbench_read_frame(input$inputs$test, categorical)
  aligned <- sdmbench_align_factors(train, test, categorical)
  train <- aligned$train
  test <- aligned$test

  train$occ <- as.integer(train$occ)
  weights <- sdmbench_pb_weights(train$occ)

  form <- if (!is.null(input$formula)) {
    stats::as.formula(as.character(input$formula))
  } else {
    build_formula(predictors, categorical, train, k = input$k)
  }

  method <- as.character(input$method %||% "REML")
  set.seed(as.integer(input$seed %||% 32639L))

  fit_start <- Sys.time()
  model <- mgcv::gam(
    form,
    data = train,
    family = stats::binomial(link = "logit"),
    weights = weights,
    method = method,
    select = isTRUE(input$select)
  )
  fit_seconds <- as.numeric(difftime(Sys.time(), fit_start, units = "secs"))

  pred_start <- Sys.time()
  preds <- as.numeric(stats::predict(model, newdata = test, type = "response"))
  predict_seconds <- as.numeric(difftime(Sys.time(), pred_start, units = "secs"))

  list(
    predictions = preds,
    fit_seconds = fit_seconds,
    predict_seconds = predict_seconds,
    model_version = as.character(utils::packageVersion("mgcv")),
    hyperparameters = list(
      formula = paste(deparse(form), collapse = " "),
      family = "binomial(logit)",
      method = method,
      basis = "tp",
      weighting = "presence_background_ratio",
      formula_source = if (is.null(input$formula)) "sdmbench_reconstructed" else "supplied"
    )
  )
}

sdmbench_main(main)
