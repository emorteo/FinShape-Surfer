#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = FALSE)
script <- sub("^--file=", "", grep("^--file=", args, value = TRUE))
root <- normalizePath(file.path(dirname(script), ".."), mustWork = TRUE)
Sys.setenv(FINFINDR_REFERENCE_ROOT = root)

source(file.path(root, "R", "reference_paths.R"))
source(file.path(root, "R", "fetch_upstream.R"))

if (!requireNamespace("renv", quietly = TRUE)) {
  install.packages("renv", repos = "https://cloud.r-project.org")
}

renv::restore(project = root, prompt = FALSE)

package_loads_in_clean_process <- function(package) {
  library_paths <- paste(deparse(.libPaths()), collapse = " ")
  expression <- sprintf(
    ".libPaths(%s); loadNamespace(%s)",
    library_paths,
    deparse(package)
  )
  status <- system2(
    file.path(R.home("bin"), "Rscript"),
    c("--vanilla", "-e", shQuote(expression)),
    stdout = FALSE,
    stderr = FALSE
  )
  identical(status, 0L)
}

if (identical(Sys.info()[["sysname"]], "Darwin")) {
  if (!nzchar(Sys.which("pkg-config"))) {
    stop("pkg-config is required. Install it with: brew install pkgconf")
  }

  required_system_packages <- c("fftw3", "libtiff-4", "x11")
  available <- vapply(
    required_system_packages,
    function(package) identical(system2("pkg-config", c("--exists", package)), 0L),
    logical(1)
  )
  if (!all(available)) {
    stop(
      "Missing macOS build dependencies: ",
      paste(required_system_packages[!available], collapse = ", "),
      ". Install them with: brew install fftw libtiff libx11 pkgconf"
    )
  }

  imager_loads <- package_loads_in_clean_process("imager")
  if (!imager_loads || identical(Sys.getenv("FINFINDR_REBUILD_IMAGER"), "1")) {
    options(configure.args = c(
      imager = "--x-includes=/opt/homebrew/include --x-libraries=/opt/homebrew/lib"
    ))
    renv::install(
      "imager",
      type = "source",
      dependencies = FALSE,
      rebuild = TRUE,
      prompt = FALSE,
      project = root
    )
  }
}

if (!package_loads_in_clean_process("imager")) {
  stop("The restored imager package cannot be loaded in the reference environment.")
}

source_dir <- fetch_finfindr_source(root)

cat("FinFindR reference source: ", source_dir, "\n", sep = "")
cat("Pinned commit: 5d7babd208af9642e440ef9143ec679c1154ff91\n")
cat("Declared package version: 0.1.21\n")
cat("Reference environment is ready. Run: Rscript scripts/verify_golden.R\n")
