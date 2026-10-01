#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = FALSE)
script <- sub("^--file=", "", grep("^--file=", args, value = TRUE))
root <- normalizePath(file.path(dirname(script), ".."), mustWork = TRUE)
Sys.setenv(FINFINDR_REFERENCE_ROOT = root)

source(file.path(root, "R", "reference_paths.R"))
source(file.path(root, "R", "fetch_upstream.R"))
source(file.path(root, "R", "load_reference.R"))
source(file.path(root, "R", "golden_cases.R"))

golden <- run_golden_cases(root)
output_dir <- file.path(root, "golden")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
output <- file.path(output_dir, "core-golden.json")
jsonlite::write_json(golden, output, auto_unbox = TRUE, pretty = TRUE, digits = NA)
cat("Wrote ", output, "\n", sep = "")
