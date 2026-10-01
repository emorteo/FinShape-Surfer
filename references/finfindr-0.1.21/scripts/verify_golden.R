#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = FALSE)
script <- sub("^--file=", "", grep("^--file=", args, value = TRUE))
root <- normalizePath(file.path(dirname(script), ".."), mustWork = TRUE)
Sys.setenv(FINFINDR_REFERENCE_ROOT = root)

source(file.path(root, "R", "reference_paths.R"))
source(file.path(root, "R", "fetch_upstream.R"))
source(file.path(root, "R", "load_reference.R"))
source(file.path(root, "R", "golden_cases.R"))

expected_path <- file.path(root, "golden", "core-golden.json")
if (!file.exists(expected_path)) stop("Missing committed golden file: ", expected_path)

expected <- jsonlite::fromJSON(expected_path, simplifyVector = FALSE)
actual <- run_golden_cases(root)
expected_json <- jsonlite::toJSON(expected, auto_unbox = TRUE, digits = NA, null = "null")
actual_json <- jsonlite::toJSON(actual, auto_unbox = TRUE, digits = NA, null = "null")

if (!identical(expected_json, actual_json)) {
  stop("FinFindR golden verification failed. Regenerate only after reviewing the upstream, toolchain, or algorithm change.")
}

cat("FinFindR 0.1.21 deterministic core golden verification passed.\n")
cat("Original R/MXNet-dependent workflows remain BLOCKED / NOT TESTED in this environment.\n")
cat("Python compatibility-layer model regression tests are run separately with pytest.\n")
