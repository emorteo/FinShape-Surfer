load_finfindr_reference <- function(root = reference_root()) {
  required <- c("digest", "imager", "jsonlite", "Rcpp")
  missing <- required[!vapply(required, requireNamespace, logical(1), quietly = TRUE)]
  if (length(missing)) {
    stop("Missing reference packages: ", paste(missing, collapse = ", "),
         ". Run scripts/bootstrap.R first.")
  }

  source_dir <- fetch_finfindr_source(root)
  build_dir <- file.path(root, ".reference", "build")
  cache_dir <- file.path(root, ".reference", "sourceCpp-cache")
  dir.create(build_dir, recursive = TRUE, showWarnings = FALSE)
  dir.create(cache_dir, recursive = TRUE, showWarnings = FALSE)

  # FinFindR's C++ algorithms are preserved byte-for-byte. Only the obsolete
  # sourceCpp dependency annotation is removed from build copies so explicit,
  # platform-neutral include flags can be supplied for modern imager builds.
  cpp_files <- c("astar.cpp", "chromaAnnulus.cpp", "imageToFeatureProcessing.cpp")
  for (name in cpp_files) {
    input <- file.path(source_dir, "src", name)
    output <- file.path(build_dir, name)
    lines <- readLines(input, warn = FALSE)
    marker <- "// [[Rcpp::depends(imager)]]"
    marker_count <- sum(lines == marker)
    expected_count <- if (name == "astar.cpp") 0L else 1L
    if (!identical(marker_count, expected_count)) {
      stop("Unexpected Rcpp dependency annotation count in ", name)
    }
    writeLines(lines[lines != marker], output, useBytes = TRUE)
  }

  old_env <- Sys.getenv(c("PKG_CPPFLAGS", "PKG_LIBS"), unset = NA_character_)
  on.exit({
    for (key in names(old_env)) {
      if (is.na(old_env[[key]])) {
        Sys.unsetenv(key)
      } else {
        do.call(Sys.setenv, stats::setNames(list(old_env[[key]]), key))
      }
    }
  }, add = TRUE)

  imager_include <- system.file("include", package = "imager")
  Sys.setenv(
    PKG_CPPFLAGS = paste0("-I", shQuote(imager_include), " -Dcimg_r_mode -fpermissive"),
    PKG_LIBS = ""
  )
  for (name in cpp_files) {
    Rcpp::sourceCpp(file.path(build_dir, name), cacheDir = cache_dir, rebuild = FALSE)
  }

  suppressPackageStartupMessages(library(imager))
  r_files <- c("pathEncoding.R", "extractFeatures.R", "compare.R", "blobber.R")
  for (name in r_files) {
    sys.source(file.path(source_dir, "R", name), envir = .GlobalEnv)
  }

  invisible(source_dir)
}
