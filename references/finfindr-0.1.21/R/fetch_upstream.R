fetch_finfindr_source <- function(root = reference_root()) {
  config <- read_reference_config(root)
  cache_root <- file.path(root, ".reference")
  source_dir <- file.path(cache_root, "upstream")
  dir.create(cache_root, recursive = TRUE, showWarnings = FALSE)

  source_exists <- dir.exists(file.path(source_dir, ".git"))
  if (!source_exists) {
    status <- system2(
      "git",
      c("clone", "--no-checkout", config$upstream_repository, source_dir)
    )
    if (!identical(status, 0L)) {
      stop("Unable to clone the pinned FinFindR upstream repository.")
    }
  }

  actual_commit <- system2(
    "git", c("-C", source_dir, "rev-parse", "HEAD"), stdout = TRUE, stderr = FALSE
  )
  dirty <- system2(
    "git", c("-C", source_dir, "status", "--porcelain"), stdout = TRUE
  )
  refresh <- identical(Sys.getenv("FINFINDR_REFRESH_UPSTREAM"), "1")
  if (identical(actual_commit, config$upstream_commit) && !length(dirty) && !refresh) {
    description <- read.dcf(file.path(source_dir, "DESCRIPTION"))
    if (!identical(unname(description[1, "Version"]), config$declared_package_version)) {
      stop("Pinned FinFindR source does not declare the expected package version.")
    }
    return(source_dir)
  }

  fetch_status <- system2(
    "git",
    c("-C", source_dir, "fetch", "--depth", "1", "origin", config$upstream_commit)
  )
  if (!identical(fetch_status, 0L)) {
    stop("Unable to fetch the pinned FinFindR commit.")
  }

  checkout_status <- system2(
    "git",
    c("-C", source_dir, "checkout", "--detach", "--force", config$upstream_commit)
  )
  if (!identical(checkout_status, 0L)) {
    stop("Unable to check out the pinned FinFindR commit.")
  }

  actual_commit <- system2(
    "git", c("-C", source_dir, "rev-parse", "HEAD"), stdout = TRUE
  )
  if (!identical(actual_commit, config$upstream_commit)) {
    stop("FinFindR source commit does not match reference.json.")
  }

  description <- read.dcf(file.path(source_dir, "DESCRIPTION"))
  if (!identical(unname(description[1, "Version"]), config$declared_package_version)) {
    stop("Pinned FinFindR source does not declare the expected package version.")
  }

  source_dir
}
