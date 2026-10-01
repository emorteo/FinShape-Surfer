reference_root <- function() {
  configured <- Sys.getenv("FINFINDR_REFERENCE_ROOT", unset = "")
  if (nzchar(configured)) {
    return(normalizePath(configured, mustWork = TRUE))
  }

  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg) == 1L) {
    script <- sub("^--file=", "", file_arg)
    return(normalizePath(file.path(dirname(script), ".."), mustWork = TRUE))
  }

  normalizePath(getwd(), mustWork = TRUE)
}

read_reference_config <- function(root = reference_root()) {
  jsonlite::fromJSON(file.path(root, "reference.json"), simplifyVector = FALSE)
}
