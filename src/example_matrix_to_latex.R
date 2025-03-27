# example_matrix_to_latex.R

# Load required package
library(xtable)

# ====================================================================
#  LOAD ENVIRONMENT VARIABLES
# ====================================================================
# Attempt to load environment variables if a .env is present
library(fs)
library(dotenv)

dotenvpath <- path(".env")
if (!file.exists(dotenvpath)) {
  dotenvpath <- path("../.env")
}
if (file.exists(dotenvpath)) {
  load_dot_env(dotenvpath)
}

# Default OUTPUT_DIR if not in environment
OUTPUT_DIR <- Sys.getenv("OUTPUT_DIR", unset = "../_output")

# Ensure OUTPUT_DIR exists
if (!dir.exists(OUTPUT_DIR)) dir.create(OUTPUT_DIR, recursive = TRUE)

# ====================================================================
#  CREATE EXAMPLE MATRIX AND CONVERT TO LATEX TABLE
# ====================================================================

# Create a simple 3x3 matrix with row/col names
mat <- matrix(1:16, nrow = 4, byrow = TRUE)
rownames(mat) <- paste0("Row", 1:4)
colnames(mat) <- paste0("Col", 1:4)

# Convert matrix to LaTeX table using xtable
table_tex <- xtable(mat)

# Write out the table as 'example_matrix_table.tex' to OUTPUT_DIR
output_file <- file.path(OUTPUT_DIR, "example_matrix_table.tex")
print(
  table_tex,
  file = output_file,
  floating = FALSE  # Don't wrap in \begin{table} ... \end{table}
)