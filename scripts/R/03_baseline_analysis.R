################################################################################
# LLM Cultural Bias — Baseline Analysis
# ============================================================================
# 1. Load data (human distributions + model results)
# 2. Compute normalized Wasserstein W1 for all 189 combinations
# 3. Bootstrap 1000× on human side → 95% CI
# 4. RQ1: ANOVA (model × country) + Tukey HSD
# 5. Identify top 5 worst-case personas (highest W1 lower CI bound)
################################################################################

library(tidyverse)

# =============================================================================
# 0. PATHS — adjust to your local repo location
# =============================================================================
REPO_DIR <- "."

human_path   <- file.path(REPO_DIR, "data/raw/human_distributions.csv")
results_dir  <- file.path(REPO_DIR, "results/baseline")
output_dir   <- file.path(REPO_DIR, "results/analysis")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# =============================================================================
# 1. LOAD DATA
# =============================================================================
cat("=" |> strrep(70), "\n")
cat("  LOADING DATA\n")
cat("=" |> strrep(70), "\n\n")

# Human distributions
human <- read_csv(human_path, show_col_types = FALSE)
cat(sprintf("  Human personas: %d\n", nrow(human)))
cat(sprintf("  Countries: %s\n", paste(sort(unique(human$country)), collapse = ", ")))

# Model results — find and load all baseline result files
model_results <- bind_rows(
  read_csv("checkpoints/baseline_gemma3.csv", show_col_types = FALSE),
  read_csv("checkpoints/baseline_bielik.csv", show_col_types = FALSE),
  read_csv("checkpoints/baseline_qwen.csv",   show_col_types = FALSE)
)

# Quick validation
model_results |>
  group_by(model) |>
  summarise(
    n_responses = n(),
    n_valid     = sum(is_valid),
    n_invalid   = sum(!is_valid),
    n_personas  = n_distinct(persona_id),
    .groups = "drop"
  ) |>
  print()

# =============================================================================
# 2. COMPUTE WASSERSTEIN W1 DISTANCE
# =============================================================================
cat("\n", "=" |> strrep(70), "\n")
cat("  COMPUTING WASSERSTEIN DISTANCES\n")
cat("=" |> strrep(70), "\n\n")

# --- Helper: W1 between two PMFs on {1, ..., 10} ---
# W1 = sum_{k=1}^{9} |CDF_a(k) - CDF_b(k)|
# Normalized to [0, 1] by dividing by 9 (max possible distance)
compute_w1 <- function(pmf_a, pmf_b) {
  cdf_a <- cumsum(pmf_a)
  cdf_b <- cumsum(pmf_b)
  # Only need k = 1..9 (CDF at k=10 is 1 for both)
  w1 <- sum(abs(cdf_a[1:9] - cdf_b[1:9]))
  return(w1)
}

compute_w1_normalized <- function(pmf_a, pmf_b) {
  compute_w1(pmf_a, pmf_b) / 9
}

# --- Build model PMFs ---
# For each persona × model, compute empirical PMF from 100 responses
model_pmfs <- model_results |>
  filter(is_valid) |>
  group_by(persona_id, model) |>
  summarise(
    n_valid = n(),
    # Count occurrences of each value 1-10
    across_values = list(table(factor(parsed_value, levels = 1:10))),
    .groups = "drop"
  ) |>
  mutate(
    pmf = map(across_values, ~ as.numeric(.x) / sum(.x))
  ) |>
  select(persona_id, model, n_valid, pmf)

# --- Build human PMFs as named list for lookup ---
human_pmf_list <- human |>
  rowwise() |>
  mutate(pmf = list(c_across(pmf_1:pmf_10))) |>
  ungroup() |>
  select(persona_id, pmf) |>
  deframe()

# --- Get human n_respondents if available ---
# Check for common column names for sample size
n_col <- intersect(names(human), c("n", "n_respondents", "N", "total_n", "count"))
if (length(n_col) > 0) {
  human_n <- human |> select(persona_id, n_resp = all_of(n_col[1])) |> deframe()
  cat(sprintf("  Found human sample sizes (column: %s)\n", n_col[1]))
  cat(sprintf("  Range: %d - %d respondents\n", min(human_n), max(human_n)))
} else {
  # If no n column, try to infer or use a default
  # Check if there's any column that could represent counts
  cat("  WARNING: No n_respondents column found. Checking for alternatives...\n")
  # Use a reasonable default based on WVS (typically 50-500 per persona)
  # The user specified N >= 10 as minimum
  human_n <- setNames(rep(100, nrow(human)), human$persona_id)
  cat("  Using default n=100 for bootstrap. Update if actual n is available.\n")
}

# --- Compute W1 for all persona × model combinations ---
w1_results <- model_pmfs |>
  rowwise() |>
  mutate(
    human_pmf = list(human_pmf_list[[persona_id]]),
    w1_raw    = compute_w1(human_pmf, pmf),
    w1_norm   = compute_w1_normalized(human_pmf, pmf)
  ) |>
  ungroup() |>
  select(persona_id, model, n_valid, w1_raw, w1_norm)

# Add demographic info
w1_results <- w1_results |>
  left_join(
    human |> select(persona_id, country, sex, age_group, education),
    by = "persona_id"
  )

# Summary statistics
cat("\n  W1 normalized summary by model:\n")
w1_results |>
  group_by(model) |>
  summarise(
    mean_w1   = mean(w1_norm),
    sd_w1     = sd(w1_norm),
    median_w1 = median(w1_norm),
    min_w1    = min(w1_norm),
    max_w1    = max(w1_norm),
    .groups   = "drop"
  ) |>
  print()

cat("\n  W1 normalized summary by model × country:\n")
w1_results |>
  group_by(model, country) |>
  summarise(
    mean_w1 = mean(w1_norm),
    sd_w1   = sd(w1_norm),
    n       = n(),
    .groups = "drop"
  ) |>
  pivot_wider(
    names_from  = country,
    values_from = c(mean_w1, sd_w1, n),
    names_glue  = "{country}_{.value}"
  ) |>
  print()

# =============================================================================
# 3. BOOTSTRAP 1000× ON HUMAN SIDE → 95% CI
# =============================================================================
cat("\n", "=" |> strrep(70), "\n")
cat("  BOOTSTRAP CONFIDENCE INTERVALS (1000 iterations)\n")
cat("=" |> strrep(70), "\n\n")

set.seed(42)
N_BOOT <- 1000

# For each persona × model:
#   - Resample human responses from PMF (multinomial)
#   - Recompute W1 with resampled human PMF vs fixed model PMF
#   - Get 95% CI
bootstrap_w1 <- function(human_pmf_vec, model_pmf_vec, n_human, n_boot = 1000) {
  boot_w1 <- numeric(n_boot)
  for (b in seq_len(n_boot)) {
    # Resample from human distribution
    boot_counts <- rmultinom(1, size = n_human, prob = human_pmf_vec)
    boot_pmf    <- as.numeric(boot_counts) / n_human
    boot_w1[b]  <- compute_w1_normalized(boot_pmf, model_pmf_vec)
  }
  return(boot_w1)
}

cat("  Running bootstrap (this may take a minute)...\n")
t_start <- Sys.time()

boot_results <- model_pmfs |>
  rowwise() |>
  mutate(
    human_pmf_vec = list(human_pmf_list[[persona_id]]),
    n_human       = human_n[[persona_id]],
    boot_dist     = list(bootstrap_w1(human_pmf_vec, pmf, n_human, N_BOOT)),
    w1_norm       = compute_w1_normalized(human_pmf_vec, pmf),
    w1_ci_lower   = quantile(boot_dist, 0.025),
    w1_ci_upper   = quantile(boot_dist, 0.975),
    w1_boot_mean  = mean(boot_dist),
    w1_boot_sd    = sd(boot_dist)
  ) |>
  ungroup() |>
  select(persona_id, model, w1_norm, w1_boot_mean, w1_boot_sd,
         w1_ci_lower, w1_ci_upper)

t_elapsed <- difftime(Sys.time(), t_start, units = "secs")
cat(sprintf("  Bootstrap completed in %.1f seconds\n", as.numeric(t_elapsed)))

# Add demographics
boot_results <- boot_results |>
  left_join(
    human |> select(persona_id, country, sex, age_group, education),
    by = "persona_id"
  )

cat("\n  Bootstrap W1 summary by model:\n")
boot_results |>
  group_by(model) |>
  summarise(
    mean_w1       = mean(w1_norm),
    mean_ci_lower = mean(w1_ci_lower),
    mean_ci_upper = mean(w1_ci_upper),
    mean_ci_width = mean(w1_ci_upper - w1_ci_lower),
    .groups       = "drop"
  ) |>
  print()

# =============================================================================
# 4. RQ1: ANOVA — Do models favor their country of origin?
# =============================================================================
cat("\n", "=" |> strrep(70), "\n")
cat("  RQ1: CULTURAL ORIGIN BIAS (ANOVA)\n")
cat("=" |> strrep(70), "\n\n")

# Model origin mapping
model_origins <- tibble(
  model        = c("gemma3", "bielik", "qwen"),
  model_origin = c("USA",    "Poland", "China")
)

rq1_data <- boot_results |>
  left_join(model_origins, by = "model") |>
  mutate(
    # Does this persona's country match the model's origin country?
    # Note: Bielik is Polish but we test on Slovakia (closest available)
    is_origin_adjacent = case_when(
      model == "gemma3" & country == "USA" ~ TRUE,
      model == "bielik" & country == "SVK" ~ TRUE,   # Slovakia = closest to Poland
      model == "qwen"   & country == "CHN" ~ TRUE,
      TRUE ~ FALSE
    )
  )

# --- Two-way ANOVA: W1 ~ model * country ---
cat("  Two-way ANOVA: w1_norm ~ model * country\n")
cat("  -----------------------------------------\n")
aov_model <- aov(w1_norm ~ model * country, data = rq1_data)
print(summary(aov_model))

# --- Tukey HSD post-hoc ---
cat("\n  Tukey HSD post-hoc tests:\n")
cat("  -------------------------\n")
tukey_result <- TukeyHSD(aov_model)
print(tukey_result)

# --- RQ1 specific test: origin alignment ---
# Compare mean W1 when model matches vs doesn't match origin country
cat("\n  Origin alignment analysis:\n")
cat("  --------------------------\n")
rq1_data |>
  group_by(model, is_origin_adjacent) |>
  summarise(
    mean_w1 = mean(w1_norm),
    sd_w1   = sd(w1_norm),
    n       = n(),
    .groups = "drop"
  ) |>
  arrange(model, is_origin_adjacent) |>
  print()

# Per-model: W1 for each country
cat("\n  Mean W1 by model × country (lower = more aligned):\n")
cat("  ---------------------------------------------------\n")
rq1_table <- rq1_data |>
  group_by(model, model_origin, country) |>
  summarise(mean_w1 = round(mean(w1_norm), 4), .groups = "drop") |>
  pivot_wider(names_from = country, values_from = mean_w1)
print(rq1_table)

cat("\n  Interpretation: If a model favors its origin country,\n")
cat("  its W1 should be LOWER for that country's personas.\n")
cat("  Gemma3 (USA): check USA column\n")
cat("  Bielik (Poland): check SVK column (proxy for Poland)\n")
cat("  Qwen (China): check CHN column\n")

# --- Effect sizes (Cohen's d) for origin vs non-origin ---
cat("\n  Cohen's d for origin alignment:\n")
cat("  --------------------------------\n")
for (m in c("gemma3", "bielik", "qwen")) {
  d <- rq1_data |> filter(model == m)
  origin     <- d |> filter(is_origin_adjacent) |> pull(w1_norm)
  non_origin <- d |> filter(!is_origin_adjacent) |> pull(w1_norm)
  
  if (length(origin) > 1 & length(non_origin) > 1) {
    pooled_sd <- sqrt(((length(origin) - 1) * var(origin) +
                         (length(non_origin) - 1) * var(non_origin)) /
                        (length(origin) + length(non_origin) - 2))
    cohens_d <- (mean(non_origin) - mean(origin)) / pooled_sd
    # t-test
    tt <- t.test(origin, non_origin)
    cat(sprintf("  %s: origin W1=%.4f, non-origin W1=%.4f, Cohen's d=%.3f, p=%.4f\n",
                m, mean(origin), mean(non_origin), cohens_d, tt$p.value))
  }
}

# =============================================================================
# 5. IDENTIFY TOP 5 WORST-CASE PERSONAS
# =============================================================================
cat("\n", "=" |> strrep(70), "\n")
cat("  TOP 5 WORST-CASE PERSONAS (for LoRA fine-tuning)\n")
cat("=" |> strrep(70), "\n\n")

# Rank by highest W1 lower confidence bound
# This ensures robust identification despite sampling variability
worst_cases <- boot_results |>
  arrange(desc(w1_ci_lower)) |>
  select(persona_id, model, country, sex, age_group, education,
         w1_norm, w1_ci_lower, w1_ci_upper)

cat("  Top 10 worst-case persona × model combinations:\n")
cat("  (ranked by W1 lower CI bound — higher = more biased)\n\n")
worst_cases |>
  head(10) |>
  print(n = 10)

# Top 5 unique personas (across all models)
cat("\n  Top 5 unique worst-case PERSONAS (aggregated across models):\n")
cat("  (these become LoRA fine-tuning targets)\n\n")
worst_personas <- boot_results |>
  group_by(persona_id, country, sex, age_group, education) |>
  summarise(
    mean_w1_norm     = mean(w1_norm),
    max_w1_norm      = max(w1_norm),
    mean_ci_lower    = mean(w1_ci_lower),
    max_ci_lower     = max(w1_ci_lower),
    worst_model      = model[which.max(w1_ci_lower)],
    worst_model_w1   = max(w1_ci_lower),
    .groups = "drop"
  ) |>
  arrange(desc(max_ci_lower))

worst_5 <- worst_personas |> head(5)
print(worst_5, n = 5)

cat("\n  These 5 personas will be targeted for LoRA fine-tuning.\n")
cat("  Fine-tuning models: Bielik + Qwen (Transformers-based)\n")

# =============================================================================
# SAVE ALL OUTPUTS
# =============================================================================
cat("\n", "=" |> strrep(70), "\n")
cat("  SAVING RESULTS\n")
cat("=" |> strrep(70), "\n\n")

# W1 results (all 189 combinations)
w1_out <- file.path(output_dir, "w1_all_combinations.csv")
w1_results |> write_csv(w1_out)
cat(sprintf("  Saved: %s\n", w1_out))

# Bootstrap results
boot_out <- file.path(output_dir, "w1_bootstrap_ci.csv")
boot_results |> write_csv(boot_out)
cat(sprintf("  Saved: %s\n", boot_out))

# RQ1 ANOVA results
rq1_out <- file.path(output_dir, "rq1_anova_summary.txt")
sink(rq1_out)
cat("Two-way ANOVA: w1_norm ~ model * country\n")
cat("==========================================\n\n")
print(summary(aov_model))
cat("\nTukey HSD:\n")
cat("==========\n\n")
print(tukey_result)
sink()
cat(sprintf("  Saved: %s\n", rq1_out))

# Worst-case personas
worst_out <- file.path(output_dir, "worst_case_personas.csv")
worst_personas |> write_csv(worst_out)
cat(sprintf("  Saved: %s\n", worst_out))

# Top 5 for LoRA
top5_out <- file.path(output_dir, "top5_lora_targets.csv")
worst_5 |> write_csv(top5_out)
cat(sprintf("  Saved: %s\n", top5_out))

# Full RQ1 table (model × country)
rq1_table_out <- file.path(output_dir, "rq1_model_country_table.csv")
rq1_table |> write_csv(rq1_table_out)
cat(sprintf("  Saved: %s\n", rq1_table_out))

cat("\n  All outputs saved to:", output_dir, "\n")
cat("=" |> strrep(70), "\n")
cat("  ANALYSIS COMPLETE\n")
cat("=" |> strrep(70), "\n")




# =============================================================================
# Top 5 worst-case personas PER MODEL
# =============================================================================
library(tidyverse)

# Load bootstrap results
boot <- read_csv("results/analysis/w1_bootstrap_ci.csv", show_col_types = FALSE)

# Top 5 per model, ranked by W1 lower CI bound
top5_per_model <- boot |>
  group_by(model) |>
  slice_max(w1_ci_lower, n = 5) |>
  arrange(model, desc(w1_ci_lower)) |>
  select(model, persona_id, country, sex, age_group, education,
         w1_norm, w1_ci_lower, w1_ci_upper) |>
  mutate(
    w1_norm     = round(w1_norm, 3),
    w1_ci_lower = round(w1_ci_lower, 3),
    w1_ci_upper = round(w1_ci_upper, 3)
  )

# Print clean table
cat("\n")
for (m in c("bielik", "gemma3", "qwen")) {
  origin <- case_when(m == "bielik" ~ "Poland", m == "gemma3" ~ "USA", m == "qwen" ~ "China")
  cat(sprintf("=== %s (%s) — Top 5 worst-case personas ===\n", toupper(m), origin))
  top5_per_model |>
    filter(model == m) |>
    ungroup() |>
    mutate(rank = row_number()) |>
    select(rank, persona_id, country, w1_norm, w1_ci_lower, w1_ci_upper) |>
    print(n = 5)
  cat("\n")
}

# Check overlaps
cat("=== OVERLAP ANALYSIS ===\n")
for (m1 in c("bielik", "gemma3", "qwen")) {
  for (m2 in c("bielik", "gemma3", "qwen")) {
    if (m1 < m2) {
      p1 <- top5_per_model |> filter(model == m1) |> pull(persona_id)
      p2 <- top5_per_model |> filter(model == m2) |> pull(persona_id)
      overlap <- intersect(p1, p2)
      cat(sprintf("  %s ∩ %s: %d overlap", m1, m2, length(overlap)))
      if (length(overlap) > 0) cat(sprintf(" → %s", paste(overlap, collapse = ", ")))
      cat("\n")
    }
  }
}

# Save
top5_per_model |> write_csv("results/analysis/top5_per_model.csv")
cat("\nSaved: results/analysis/top5_per_model.csv\n")