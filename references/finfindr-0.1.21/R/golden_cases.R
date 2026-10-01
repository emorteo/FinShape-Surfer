canonical_numeric <- function(value, digits = 12L) {
  rounded <- round(as.numeric(value), digits)
  payload <- jsonlite::toJSON(rounded, auto_unbox = TRUE, digits = NA, na = "string")
  list(
    dimensions = unname(dim(value)),
    length = length(value),
    minimum = min(rounded, na.rm = TRUE),
    maximum = max(rounded, na.rm = TRUE),
    mean = mean(rounded, na.rm = TRUE),
    sha256 = digest::digest(payload, algo = "sha256", serialize = FALSE)
  )
}

synthetic_fin_image <- function() {
  width <- 180L
  height <- 160L
  image <- array(0, dim = c(width, height, 1L, 3L))
  for (channel in seq_len(3L)) {
    for (y in seq_len(height)) {
      image[, y, 1L, channel] <- (
        (seq_len(width) - 1L) / (width - 1L) * 0.45 +
        (y - 1L) / (height - 1L) * 0.35 +
        (channel - 1L) * 0.10
      )
    }
  }
  imager::as.cimg(image)
}

run_golden_cases <- function(root = reference_root()) {
  config <- read_reference_config(root)
  source_dir <- load_finfindr_reference(root)

  path <- cbind(
    x = c(10L, 11L, 12L, 13L, 14L, 14L, 15L, 16L),
    y = c(20L, 20L, 21L, 22L, 22L, 23L, 24L, 24L)
  )
  encoded <- encodePath(path)
  decoded <- suppressMessages(decodePath(encoded))

  query <- c(1.0, 2.0, 3.0, 4.0)
  reference <- cbind(
    identical = query,
    offset = c(2.0, 4.0, 6.0, 8.0),
    reflected = rev(query)
  )
  distances <- distanceToRef(query, reference)

  edge <- matrix(0.02, nrow = 40L, ncol = 40L)
  for (index in 3:36) {
    edge[index, index] <- 1.0
    edge[index, index + 1L] <- 0.85
  }
  found_path <- findPath(edge, 3L, 3L, 36L, 36L, 1L, 38L, 1L, 38L, 2L)

  gradient <- array(0, dim = c(15L, 15L, 1L, 1L))
  angles <- array(0, dim = dim(gradient))
  gradient[8L, 3:13, 1L, 1L] <- seq(0.2, 1.0, length.out = 11L)
  gradient[7L, 3:13, 1L, 1L] <- gradient[8L, 3:13, 1L, 1L] * 0.4
  gradient[9L, 3:13, 1L, 1L] <- gradient[8L, 3:13, 1L, 1L] * 0.3
  edge_map <- extractEdgeMap(imager::as.cimg(gradient), imager::as.cimg(angles))
  edge_map_summary <- canonical_numeric(edge_map)
  edge_map_summary$nonzero_pixels <- sum(abs(as.numeric(edge_map)) > 0)

  image <- synthetic_fin_image()
  trace_x <- 20:159
  trace_y <- as.integer(round(80 + 20 * sin(seq(0, pi, length.out = length(trace_x)))))
  annulus <- extractAnnulus(image, trace_x, trace_y)

  glare_image <- image
  glare_coords <- expand.grid(x = 80:82, y = 70:72, cc = 0:2)
  for (row in seq_len(nrow(glare_coords))) {
    glare_image[
      glare_coords$x[row] + 1L,
      glare_coords$y[row] + 1L,
      1L,
      glare_coords$cc[row] + 1L
    ] <- 0
  }
  filled <- fillGlare(glare_image, glare_coords, 7L)
  filled_summary <- canonical_numeric(filled)
  filled_summary$zero_pixels_before <- sum(as.numeric(glare_image) == 0)
  filled_summary$zero_pixels_after <- sum(as.numeric(filled) == 0)
  filled_summary$changed_pixels <- sum(abs(as.numeric(filled) - as.numeric(glare_image)) > 1e-12)

  resized <- constrainSizeFinImage(image, maxDim = 120L, minDim = 60L)

  if (!isTRUE(all.equal(unname(decoded), unname(path), check.attributes = FALSE))) {
    stop("Path encoding golden case did not round-trip exactly.")
  }
  if (!length(found_path) || any(!found_path %in% 0:7)) {
    stop("A* golden case did not return a valid direction sequence.")
  }
  if (edge_map_summary$nonzero_pixels <= 0L) {
    stop("Edge-map golden case produced no detected edge pixels.")
  }
  if (filled_summary$changed_pixels <= 0L ||
      filled_summary$zero_pixels_after >= filled_summary$zero_pixels_before) {
    stop("Glare-fill golden case did not replace the synthetic glare pixels.")
  }
  numeric_outputs <- list(distances, edge_map, annulus, filled, resized)
  if (!all(vapply(numeric_outputs, function(value) all(is.finite(value)), logical(1)))) {
    stop("A golden case produced a non-finite numeric value.")
  }
  resized_dimensions <- dim(resized)[1:2]
  if (max(resized_dimensions) > 120L || min(resized_dimensions) < 60L) {
    stop("Bounded resize golden case violated its requested dimensions.")
  }

  source_hashes <- vapply(
    c("DESCRIPTION", "R/pathEncoding.R", "R/extractFeatures.R", "R/compare.R",
      "R/blobber.R", "src/astar.cpp", "src/chromaAnnulus.cpp",
      "src/imageToFeatureProcessing.cpp"),
    function(relative) digest::digest(file.path(source_dir, relative), algo = "sha256", file = TRUE),
    character(1)
  )

  list(
    schema_version = 1L,
    upstream = list(
      repository = config$upstream_repository,
      commit = config$upstream_commit,
      declared_version = config$declared_package_version,
      source_sha256 = as.list(source_hashes)
    ),
    environment = list(
      r_version = paste(R.version$major, R.version$minor, sep = "."),
      platform = R.version$platform,
      packages = as.list(vapply(
        c("Rcpp", "imager", "jsonlite", "digest", "renv"),
        function(package) as.character(utils::packageVersion(package)),
        character(1)
      ))
    ),
    coverage = list(
      verified = c("path encoding", "distance comparison", "A* path search",
                   "non-maximum edge suppression", "annulus extraction",
                   "glare filling", "bounded image resize"),
      blocked = config$excluded_scope
    ),
    results = list(
      path_encoding = list(
        encoded = as.integer(encoded),
        decoded = lapply(seq_len(nrow(decoded)), function(index) as.integer(decoded[index, ])),
        exact_roundtrip = isTRUE(all.equal(unname(decoded), unname(path), check.attributes = FALSE))
      ),
      distance_to_reference = as.list(unname(distances)),
      astar_path = list(directions = as.integer(found_path), length = length(found_path)),
      edge_map = edge_map_summary,
      annulus = canonical_numeric(annulus),
      filled_glare = filled_summary,
      resized_image = canonical_numeric(resized)
    )
  )
}
