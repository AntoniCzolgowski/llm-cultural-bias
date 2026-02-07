# =============================================================================
# data_preparation.R — Human side of W1 pipeline
# Loads WVS Q164, encodes personas, computes PMFs, saves CSV.
#
# Requirements: tidyverse
#   install.packages("tidyverse")
# =============================================================================

library(tidyverse)

# --- Paths -------------------------------------------------------------------
input_path  <- "C:/Users/Antoni/OneDrive/Pulpit/CuBoulder/2 sem/Study/github/llm-cultural-bias/data/raw/test.csv"
output_dir  <- "C:/Users/Antoni/OneDrive/Pulpit/CuBoulder/2 sem/Study/Rstudio/llm-cultural-bias/data/raw"

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# --- Load & Validate ---------------------------------------------------------
cat("Loading WVS data...\n")
df <- read_csv(input_path, show_col_types = FALSE, locale = locale(encoding = "UTF-8"))

# Handle BOM if present (Windows UTF-8 files)
names(df) <- gsub("^\uFEFF", "", names(df))

stopifnot(
  all(c("B_COUNTRY_ALPHA", "Q164", "Q260", "Q262", "Q275R") %in% names(df)),
  all(df$Q164 >= 1 & df$Q164 <= 10),
  sum(is.na(df)) == 0
)
cat(sprintf("  %d respondents, %d countries\n", nrow(df), n_distinct(df$B_COUNTRY_ALPHA)))

# --- Encode Personas ---------------------------------------------------------
cat("Encoding personas...\n")

df <- df %>%
  mutate(
    country   = B_COUNTRY_ALPHA,
    sex       = if_else(Q260 == 1, "Male", "Female"),
    age_group = case_when(
      Q262 <= 29 ~ "18-29",
      Q262 <= 49 ~ "30-49",
      Q262 <= 64 ~ "50-64",
      TRUE       ~ "65+"
    ),
    education = case_when(
      Q275R == 1 ~ "Lower",
      Q275R == 2 ~ "Medium",
      Q275R == 3 ~ "Higher"
    ),
    persona_id = paste(country, sex, age_group, education, sep = "_")
  )

cat(sprintf("  %d total personas\n", n_distinct(df$persona_id)))

# --- Compute PMFs (N >= 10) --------------------------------------------------
cat("Computing PMFs (N >= 10)...\n")

compute_pmf <- function(responses) {
  counts <- tabulate(responses, nbins = 10)
  counts / sum(counts)
}

personas <- df %>%
  group_by(persona_id, country, sex, age_group, education) %>%
  summarise(
    n_respondents = n(),
    mean_response = mean(Q164),
    std_response  = sd(Q164),
    pmf           = list(compute_pmf(Q164)),
    .groups = "drop"
  ) %>%
  filter(n_respondents >= 10) %>%
  arrange(persona_id)

cat(sprintf("  %d valid personas\n", nrow(personas)))

# --- Summary -----------------------------------------------------------------
cat("\n--- Summary by Country ---\n")
for (c in c("CHN", "SVK", "USA")) {
  sub <- personas %>% filter(country == c)
  cat(sprintf("  %s: %d personas, N range [%d-%d], mean response %.2f\n",
              c, nrow(sub), as.integer(min(sub$n_respondents)),
              as.integer(max(sub$n_respondents)),
              mean(sub$mean_response)))
}

cat(sprintf("\n--- Sample Size Tiers ---\n"))
cat(sprintf("  N < 30:        %d personas (wider CIs expected)\n",
            sum(personas$n_respondents < 30)))
cat(sprintf("  30 <= N < 100: %d personas\n",
            sum(personas$n_respondents >= 30 & personas$n_respondents < 100)))
cat(sprintf("  N >= 100:      %d personas\n",
            sum(personas$n_respondents >= 100)))

# --- Save CSV ----------------------------------------------------------------
csv_df <- personas %>%
  select(persona_id, country, sex, age_group, education,
         n_respondents, mean_response, std_response) %>%
  bind_cols(
    personas$pmf %>%
      map(~ set_names(.x, paste0("pmf_", 1:10))) %>%
      bind_rows()
  )

csv_path <- file.path(output_dir, "human_distributions.csv")
write_csv(csv_df, csv_path)
cat(sprintf("\nSaved: %s\n", csv_path))

# --- Print Full Table --------------------------------------------------------
cat(sprintf("\n--- All %d Personas ---\n", nrow(personas)))
cat(sprintf("%-35s %5s %6s %6s\n", "persona_id", "N", "mean", "std"))
cat(strrep("-", 55), "\n")
for (i in seq_len(nrow(personas))) {
  r <- personas[i, ]
  cat(sprintf("%-35s %5d %6.2f %6.2f\n",
              r$persona_id, as.integer(r$n_respondents),
              r$mean_response, r$std_response))
}

cat("\nDone.\n")