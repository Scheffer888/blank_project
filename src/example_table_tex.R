# example_data_generator.R

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

data_matrix <- matrix(round(runif(12, 1, 100)), nrow = 4)
colnames(data_matrix) <- c("Var1", "Var2", "Var3")

# Convert matrix to LaTeX table using xtable
table_tex <- xtable(data_matrix)

output_file <- file.path(OUTPUT_DIR, "example_table.tex")

print(
  table_tex,
  file = output_file,
  floating = FALSE  # Don't wrap in \begin{table} ... \end{table}
)