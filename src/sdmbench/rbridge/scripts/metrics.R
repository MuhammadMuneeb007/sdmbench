# metrics.R -- reference metric implementations in R, for cross-language parity.
#
# Dinnage & Warren (2026) sec. 2.6 computed ROC-AUC and PR-AUC with the
# `yardstick` package. sdmbench computes metrics in Python (scikit-learn), which
# is fast and dependency-light, but "our AUC agrees with theirs" is a claim that
# has to be *tested*, not assumed -- yardstick and scikit-learn differ in tie
# handling and in the PR-curve interpolation rule.
#
# This script is the R side of that parity test. It also exposes
# tidysdm::boyce_cont() as the reference implementation of the Continuous Boyce
# Index, which sdmbench reimplements in Python.
#
# Parity tests live in tests/test_metric_parity_r.py and are marked
# `@pytest.mark.r` -- they are skipped unless R and the packages are present.

.script_dir <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg) > 0L) dirname(sub("^--file=", "", file_arg[[1L]])) else getwd()
}
source(file.path(.script_dir(), "common.R"))

`%||%` <- function(a, b) if (is.null(a)) b else a

# Miller's calibration slope, exactly as described in the paper (sec. 2.6):
#
#   "Miller's calibration slope and intercept are obtained by regressing
#    observed outcomes on the logit-transformed predicted probabilities."
#
# i.e. glm(observed ~ logit(p), family = binomial). The slope is the
# coefficient on logit(p); a value of 1.0 means predicted odds ratios match
# observed odds ratios. This is NOT the same as a generic "calibration slope"
# from a linear regression of outcomes on probabilities.
miller_calibration <- function(truth, prob, eps = 1e-6) {
  p <- pmin(pmax(as.numeric(prob), eps), 1 - eps)
  lp <- log(p / (1 - p))
  fit <- stats::glm(as.integer(truth) ~ lp, family = stats::binomial(link = "logit"))
  co <- stats::coef(fit)
  list(
    calibration_slope = unname(co[["lp"]]),
    calibration_intercept = unname(co[["(Intercept)"]])
  )
}

main <- function(input) {
  sdmbench_require("jsonlite")

  truth <- as.integer(input$truth)
  prob <- as.numeric(input$prob)
  if (length(truth) != length(prob)) stop("truth and prob must have equal length")

  which_metrics <- if (is.null(input$metrics)) {
    c("roc_auc", "pr_auc", "calibration", "boyce")
  } else {
    as.character(input$metrics)
  }

  out <- list(n = length(truth))

  if (any(c("roc_auc", "pr_auc") %in% which_metrics)) {
    sdmbench_require("yardstick")
    # yardstick expects a factor truth with the event level FIRST, and the
    # probability column for that event level.
    df <- data.frame(
      truth = factor(ifelse(truth == 1L, "yes", "no"), levels = c("yes", "no")),
      prob = prob
    )
    if ("roc_auc" %in% which_metrics) {
      out$roc_auc <- yardstick::roc_auc_vec(df$truth, df$prob)
    }
    if ("pr_auc" %in% which_metrics) {
      out$pr_auc <- yardstick::pr_auc_vec(df$truth, df$prob)
    }
    out$yardstick_version <- as.character(utils::packageVersion("yardstick"))
  }

  if ("calibration" %in% which_metrics) {
    cal <- miller_calibration(truth, prob)
    out$calibration_slope <- cal$calibration_slope
    out$calibration_intercept <- cal$calibration_intercept
  }

  if ("boyce" %in% which_metrics) {
    # tidysdm::boyce_cont() is the established R implementation sdmbench's
    # Python Boyce index is validated against.
    if (requireNamespace("tidysdm", quietly = TRUE)) {
      df <- data.frame(
        truth = factor(ifelse(truth == 1L, "presence", "absence"),
                       levels = c("presence", "absence")),
        prob = prob
      )
      out$boyce <- tryCatch(
        as.numeric(tidysdm::boyce_cont_vec(df$truth, df$prob)),
        error = function(e) NA_real_
      )
      out$tidysdm_version <- as.character(utils::packageVersion("tidysdm"))
    } else {
      out$boyce <- NA_real_
      out$boyce_note <- "tidysdm not installed"
    }
  }

  out
}

sdmbench_main(main)
