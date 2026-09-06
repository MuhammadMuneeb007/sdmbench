# fetch_disdat.R -- export the `disdat` benchmark data to a portable format.
#
# `disdat` ships its data as .rds files inside the installed R package, which
# Python cannot read directly. This script converts every region into
# long-format CSV files with a stable schema, plus a metadata.json describing
# regions, groups, species, predictors and coordinate reference systems.
#
# Output schema (one CSV per region and table type):
#
#   <REGION>_po.csv   region, group, siteid, spid, x, y, occ, <predictors...>
#   <REGION>_bg.csv   region, group, siteid, spid, x, y, occ, <predictors...>
#   <REGION>_pa.csv   region, group, siteid, x, y, spid, occ      (long format)
#   <REGION>_env.csv  region, group, siteid, x, y, <predictors...>
#
# The `pa` table is reshaped from disdat's wide format (one column per species)
# into long format so that a species task is a simple filter on `spid`.
#
# This script only READS the installed disdat package. It never downloads,
# installs or modifies anything.

# `sys.frame()$ofile` is unavailable under Rscript; resolve the script directory
# from the command line instead so `common.R` is found reliably.
.script_dir <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg) > 0L) {
    return(dirname(sub("^--file=", "", file_arg[[1L]])))
  }
  getwd()
}
source(file.path(.script_dir(), "common.R"))

ALL_REGIONS <- c("AWT", "CAN", "NSW", "NZ", "SA", "SWI")

# Regions with more than one survey group. For these, disPa()/disEnv() must be
# called per group because the test surveys differ between groups.
GROUPS <- list(
  AWT = c("bird", "plant"),
  NSW = c("ba", "db", "nb", "ot", "ou", "rt", "ru", "sr")
)

# The five categorical predictors across the whole benchmark. Source: the
# disdat "Modeling NCEAS data" vignette, and Dinnage & Warren (2026) sec. 2.1.1,
# which agree exactly.
CATEGORICAL_VARS <- c("ontveg", "vegsys", "toxicats", "age", "calc")

standardise_po_bg <- function(df, region, kind) {
  # disPo/disBg return: siteid, spid (po only), x, y, occ, group, <predictors>.
  # Column names vary slightly in order between regions, so select by name.
  out <- data.frame(region = region, stringsAsFactors = FALSE)
  out$group <- if ("group" %in% names(df)) as.character(df$group) else NA_character_
  out$siteid <- if ("siteid" %in% names(df)) as.character(df$siteid) else seq_len(nrow(df))
  out$spid <- if ("spid" %in% names(df)) as.character(df$spid) else NA_character_
  out$x <- as.numeric(df$x)
  out$y <- as.numeric(df$y)
  out$occ <- if ("occ" %in% names(df)) as.integer(df$occ) else if (kind == "po") 1L else 0L
  meta <- c("group", "siteid", "spid", "x", "y", "occ")
  preds <- setdiff(names(df), meta)
  cbind(out, df[, preds, drop = FALSE])
}

standardise_env <- function(df, region, group) {
  out <- data.frame(region = region, stringsAsFactors = FALSE)
  out$group <- if ("group" %in% names(df)) as.character(df$group) else group
  out$siteid <- if ("siteid" %in% names(df)) as.character(df$siteid) else seq_len(nrow(df))
  out$x <- as.numeric(df$x)
  out$y <- as.numeric(df$y)
  meta <- c("group", "siteid", "spid", "x", "y", "occ")
  preds <- setdiff(names(df), meta)
  cbind(out, df[, preds, drop = FALSE])
}

# Reshape the wide presence-absence table (one column per species) to long
# format. Mirrors disdat:::.reshape_pa but keyed on names rather than position.
reshape_pa_long <- function(df, region, group) {
  meta <- intersect(c("group", "siteid", "x", "y"), names(df))
  species_cols <- setdiff(names(df), c(meta, "spid", "occ"))
  if (length(species_cols) == 0L) {
    stop(sprintf("no species columns found in pa table for %s/%s", region, group))
  }
  pieces <- lapply(species_cols, function(sp) {
    out <- data.frame(region = region, stringsAsFactors = FALSE)
    out$group <- if ("group" %in% names(df)) as.character(df$group) else group
    out$siteid <- if ("siteid" %in% names(df)) as.character(df$siteid) else seq_len(nrow(df))
    out$x <- as.numeric(df$x)
    out$y <- as.numeric(df$y)
    out$spid <- sp
    out$occ <- as.integer(df[[sp]])
    out
  })
  do.call(rbind, pieces)
}

main <- function(input) {
  sdmbench_require(c("disdat", "jsonlite"))

  outdir <- input$output_dir
  if (is.null(outdir)) stop("output_dir is required")
  dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

  regions <- input$regions
  if (is.null(regions) || length(regions) == 0L) regions <- ALL_REGIONS
  regions <- toupper(as.character(regions))

  meta <- list(
    disdat_version = as.character(utils::packageVersion("disdat")),
    r_version = paste(R.version$major, R.version$minor, sep = "."),
    exported_at = format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z"),
    categorical_variables = CATEGORICAL_VARS,
    regions = list()
  )

  for (region in regions) {
    po <- disdat::disPo(region)
    bg <- disdat::disBg(region)
    predictors <- disdat::disPredictors(region)

    po_std <- standardise_po_bg(po, region, "po")
    bg_std <- standardise_po_bg(bg, region, "bg")

    utils::write.csv(po_std, file.path(outdir, paste0(region, "_po.csv")),
                     row.names = FALSE, na = "NA")
    utils::write.csv(bg_std, file.path(outdir, paste0(region, "_bg.csv")),
                     row.names = FALSE, na = "NA")

    groups <- GROUPS[[region]]
    pa_all <- list()
    env_all <- list()
    if (is.null(groups)) {
      pa_all[[1L]] <- reshape_pa_long(disdat::disPa(region), region, NA_character_)
      env_all[[1L]] <- standardise_env(disdat::disEnv(region), region, NA_character_)
    } else {
      for (g in groups) {
        pa_all[[g]] <- reshape_pa_long(disdat::disPa(region, g), region, g)
        env_all[[g]] <- standardise_env(disdat::disEnv(region, g), region, g)
      }
    }
    pa_std <- do.call(rbind, pa_all)
    env_std <- do.call(rbind, env_all)

    utils::write.csv(pa_std, file.path(outdir, paste0(region, "_pa.csv")),
                     row.names = FALSE, na = "NA")
    utils::write.csv(env_std, file.path(outdir, paste0(region, "_env.csv")),
                     row.names = FALSE, na = "NA")

    species <- sort(unique(as.character(po_std$spid)))
    # Map each species to its survey group so the correct test survey is used.
    species_group <- vapply(
      species,
      function(s) {
        g <- unique(as.character(po_std$group[po_std$spid == s]))
        g <- g[!is.na(g)]
        if (length(g) == 0L) NA_character_ else g[[1L]]
      },
      character(1)
    )

    meta$regions[[region]] <- list(
      region = region,
      crs_proj4 = disdat::disCRS(region, "proj4"),
      crs_epsg = disdat::disCRS(region, "epsg"),
      predictors = predictors,
      categorical_predictors = intersect(CATEGORICAL_VARS, predictors),
      groups = if (is.null(groups)) list() else as.list(groups),
      n_species = length(species),
      species = as.list(species),
      species_group = as.list(species_group),
      n_presence_records = nrow(po_std),
      n_background_records = nrow(bg_std),
      n_test_sites = nrow(env_std)
    )
  }

  meta$total_species <- sum(vapply(meta$regions, function(r) r$n_species, numeric(1)))
  sdmbench_write_output(meta, file.path(outdir, "metadata.json"))

  list(
    status = "ok",
    output_dir = outdir,
    regions = regions,
    total_species = meta$total_species,
    disdat_version = meta$disdat_version,
    metadata_path = file.path(outdir, "metadata.json")
  )
}

sdmbench_main(main)
