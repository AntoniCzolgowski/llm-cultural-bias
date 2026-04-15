################################################################################
# LLM Cultural Bias — Post-Evaluation Analysis (RQ2 + RQ3)
# ============================================================================
# RQ2: Does LoRA fine-tuning on worst-case personas reduce bias?
#       → Paired t-test on 5 worst personas per model (before vs after W1)
# RQ3: Are there side effects on other personas?
#       → Paired t-test on remaining 58 personas per model
#       → Bonferroni correction for 4 tests
#
# Input:  human_distributions.csv, w1_bootstrap_ci.csv, top5_per_model.csv,
#         qwen_lora_responses.csv, bielik_lora_responses.csv
# Output: results/analysis/rq2_*.csv, rq3_*.csv, posteval_*.csv
################################################################################

library(tidyverse)

# =============================================================================
# 0. PATHS — adjust to your local repo location
# =============================================================================
REPO_DIR <- "."

human_path     <- file.path(REPO_DIR, "data/raw/human_distributions.csv")
baseline_w1    <- file.path(REPO_DIR, "results/analysis/w1_bootstrap_ci.csv")
top5_path      <- file.path(REPO_DIR, "results/analysis/top5_per_model.csv")
posteval_dir   <- file.path(REPO_DIR, "results/posteval")
output_dir     <- file.path(REPO_DIR, "results/analysis_posteval")  # ← JEDYNY output_dir
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

N_BOOT <- 1000
set.seed(42)

# =============================================================================
# 1. LOAD DATA
# =============================================================================
cat("=" |> strrep(70), "\n")
cat("  LOADING DATA\n")
cat("=" |> strrep(70), "\n\n")

# Human distributions
human <- read_csv(human_path, show_col_types = FALSE)
cat(sprintf("  Human personas: %d\n", nrow(human)))

# Baseline W1 (with bootstrap CI)
baseline <- read_csv(baseline_w1, show_col_types = FALSE)
cat(sprintf("  Baseline W1 combinations: %d\n", nrow(baseline)))

# Top 5 worst per model
top5 <- read_csv(top5_path, show_col_types = FALSE)
cat(sprintf("  Top 5 targets loaded: %d rows\n", nrow(top5)))
cat("  Bielik targets:", top5 |> filter(model == "bielik") |> pull(persona_id) |> paste(collapse = ", "), "\n")
cat("  Qwen targets:  ", top5 |> filter(model == "qwen")   |> pull(persona_id) |> paste(collapse = ", "), "\n")

# Post-eval responses
qwen_post  <- read_csv(file.path(posteval_dir, "qwen_lora_responses.csv"), show_col_types = FALSE)
bielik_post <- read_csv(file.path(posteval_dir, "bielik_lora_responses.csv"), show_col_types = FALSE)
cat(sprintf("  Qwen+LoRA responses: %d\n", nrow(qwen_post)))
cat(sprintf("  Bielik+LoRA responses: %d\n", nrow(bielik_post)))

posteval_all <- bind_rows(qwen_post, bielik_post)

# =============================================================================
# 2. COMPUTE POSTEVAL W1 DISTANCES
# =============================================================================
cat("\n", "=" |> strrep(70), "\n")
cat("  COMPUTING POST-EVAL WASSERSTEIN DISTANCES\n")
cat("=" |> strrep(70), "\n\n")

# --- W1 helper (same as baseline script) ---
compute_w1 <- function(pmf_a, pmf_b) {
  cdf_a <- cumsum(pmf_a)
  cdf_b <- cumsum(pmf_b)
  sum(abs(cdf_a[1:9] - cdf_b[1:9]))
}

compute_w1_norm <- function(pmf_a, pmf_b) {
  compute_w1(pmf_a, pmf_b) / 9
}

# --- Build human PMFs ---
human_pmf_list <- human |>
  rowwise() |>
  mutate(pmf = list(c_across(pmf_1:pmf_10))) |>
  ungroup() |>
  select(persona_id, pmf) |>
  deframe()

human_n <- human |> select(persona_id, n_respondents) |> deframe()

# --- Build posteval model PMFs ---
posteval_pmfs <- posteval_all |>
  filter(!is.na(parsed_value)) |>
  group_by(persona_id, model) |>
  summarise(
    n_valid = n(),
    counts = list(table(factor(parsed_value, levels = 1:10))),
    .groups = "drop"
  ) |>
  mutate(
    pmf = map(counts, ~ as.numeric(.x) / sum(.x))
  )

# --- Compute W1 for all posteval combinations ---
posteval_w1 <- posteval_pmfs |>
  rowwise() |>
  mutate(
    human_pmf = list(human_pmf_list[[persona_id]]),
    w1_norm   = compute_w1_norm(human_pmf, pmf)
  ) |>
  ungroup() |>
  select(persona_id, model, n_valid, w1_norm)

# Add demographics
posteval_w1 <- posteval_w1 |>
  left_join(
    human |> select(persona_id, country, sex, age_group, education),
    by = "persona_id"
  )

cat("  Post-eval W1 summary by model:\n")
posteval_w1 |>
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

# =============================================================================
# 3. BOOTSTRAP 1000× ON HUMAN SIDE FOR POSTEVAL W1
# =============================================================================
cat("\n", "=" |> strrep(70), "\n")
cat("  BOOTSTRAP CI FOR POST-EVAL W1\n")
cat("=" |> strrep(70), "\n\n")

bootstrap_w1 <- function(human_pmf_vec, model_pmf_vec, n_human, n_boot = 1000) {
  boot_w1 <- numeric(n_boot)
  for (b in seq_len(n_boot)) {
    boot_counts <- rmultinom(1, size = n_human, prob = human_pmf_vec)
    boot_pmf    <- as.numeric(boot_counts) / n_human
    boot_w1[b]  <- compute_w1_norm(boot_pmf, model_pmf_vec)
  }
  return(boot_w1)
}

cat("  Running bootstrap (this may take ~30 seconds)...\n")
t_start <- Sys.time()

posteval_boot <- posteval_pmfs |>
  rowwise() |>
  mutate(
    human_pmf_vec = list(human_pmf_list[[persona_id]]),
    n_human       = human_n[[persona_id]],
    boot_dist     = list(bootstrap_w1(human_pmf_vec, pmf, n_human, N_BOOT)),
    w1_norm       = compute_w1_norm(human_pmf_vec, pmf),
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
posteval_boot <- posteval_boot |>
  left_join(
    human |> select(persona_id, country, sex, age_group, education),
    by = "persona_id"
  )

# =============================================================================
# 4. BUILD COMPARISON TABLE: BASELINE vs POSTEVAL
# =============================================================================
cat("\n", "=" |> strrep(70), "\n")
cat("  BUILDING BASELINE vs POSTEVAL COMPARISON\n")
cat("=" |> strrep(70), "\n\n")

# Map posteval model names to baseline model names
model_map <- c("bielik_lora" = "bielik", "qwen_lora" = "qwen")

comparison <- posteval_boot |>
  mutate(base_model = model_map[model]) |>
  rename_with(~ paste0("post_", .x), c(w1_norm, w1_boot_mean, w1_boot_sd, w1_ci_lower, w1_ci_upper)) |>
  left_join(
    baseline |>
      filter(model %in% c("bielik", "qwen")) |>
      select(persona_id, model, w1_norm, w1_ci_lower, w1_ci_upper) |>
      rename(base_w1 = w1_norm, base_ci_lower = w1_ci_lower, base_ci_upper = w1_ci_upper),
    by = c("persona_id", "base_model" = "model")
  ) |>
  mutate(
    delta_w1 = post_w1_norm - base_w1,  # negative = improvement
    pct_change = (delta_w1 / base_w1) * 100
  )

cat("  Overall comparison (negative delta = bias reduced):\n")
comparison |>
  group_by(model) |>
  summarise(
    n_personas     = n(),
    mean_base_w1   = mean(base_w1),
    mean_post_w1   = mean(post_w1_norm),
    mean_delta     = mean(delta_w1),
    mean_pct_change = mean(pct_change),
    n_improved     = sum(delta_w1 < 0),
    n_worsened     = sum(delta_w1 > 0),
    n_unchanged    = sum(delta_w1 == 0),
    .groups = "drop"
  ) |>
  print()

# =============================================================================
# 5. RQ2: DOES LORA REDUCE BIAS ON WORST-CASE PERSONAS?
# =============================================================================
cat("\n", "=" |> strrep(70), "\n")
cat("  RQ2: LORA EFFECTIVENESS ON WORST-CASE PERSONAS\n")
cat("=" |> strrep(70), "\n\n")

# --- Per-model analysis ---
rq2_results <- list()

for (m in c("bielik", "qwen")) {
  m_lora <- paste0(m, "_lora")
  targets <- top5 |> filter(model == m) |> pull(persona_id)
  
  cat(sprintf("  --- %s (targets: %s) ---\n", toupper(m), paste(targets, collapse = ", ")))
  
  # Get before/after W1 for target personas
  before <- baseline |>
    filter(model == m, persona_id %in% targets) |>
    select(persona_id, w1_norm) |>
    rename(w1_before = w1_norm) |>
    arrange(persona_id)
  
  after <- posteval_boot |>
    filter(model == m_lora, persona_id %in% targets) |>
    select(persona_id, w1_norm) |>
    rename(w1_after = w1_norm) |>
    arrange(persona_id)
  
  paired <- inner_join(before, after, by = "persona_id") |>
    mutate(
      delta = w1_after - w1_before,
      pct_change = (delta / w1_before) * 100
    )
  
  cat("\n  Per-persona comparison:\n")
  print(paired, n = 5)
  
  # Paired t-test
  tt <- t.test(paired$w1_after, paired$w1_before, paired = TRUE)
  
  # Cohen's d for paired data
  d_values <- paired$delta
  cohens_d <- mean(d_values) / sd(d_values)
  
  cat(sprintf("\n  Paired t-test: t=%.3f, df=%d, p=%.6f\n",
              tt$statistic, tt$parameter, tt$p.value))
  cat(sprintf("  Mean delta: %.4f (%.1f%%)\n", mean(d_values), mean(paired$pct_change)))
  cat(sprintf("  Cohen's d: %.3f\n", cohens_d))
  cat(sprintf("  95%% CI of mean difference: [%.4f, %.4f]\n",
              tt$conf.int[1], tt$conf.int[2]))
  
  direction <- if (mean(d_values) < 0) "REDUCED" else "INCREASED"
  cat(sprintf("  Interpretation: Bias %s after LoRA fine-tuning\n\n", direction))
  
  rq2_results[[m]] <- list(
    model = m,
    paired_data = paired,
    t_stat = tt$statistic,
    df = tt$parameter,
    p_value = tt$p.value,
    mean_delta = mean(d_values),
    cohens_d = cohens_d,
    ci_lower = tt$conf.int[1],
    ci_upper = tt$conf.int[2]
  )
}

# =============================================================================
# 6. RQ3: SIDE EFFECTS ON NON-TARGET PERSONAS
# =============================================================================
cat("=" |> strrep(70), "\n")
cat("  RQ3: SIDE EFFECTS ON NON-TARGET PERSONAS\n")
cat("=" |> strrep(70), "\n\n")

rq3_results <- list()

for (m in c("bielik", "qwen")) {
  m_lora <- paste0(m, "_lora")
  targets <- top5 |> filter(model == m) |> pull(persona_id)
  non_targets <- setdiff(human$persona_id, targets)
  
  cat(sprintf("  --- %s (non-target personas: %d) ---\n", toupper(m), length(non_targets)))
  
  before <- baseline |>
    filter(model == m, persona_id %in% non_targets) |>
    select(persona_id, w1_norm) |>
    rename(w1_before = w1_norm) |>
    arrange(persona_id)
  
  after <- posteval_boot |>
    filter(model == m_lora, persona_id %in% non_targets) |>
    select(persona_id, w1_norm) |>
    rename(w1_after = w1_norm) |>
    arrange(persona_id)
  
  paired <- inner_join(before, after, by = "persona_id") |>
    mutate(
      delta = w1_after - w1_before,
      pct_change = (delta / w1_before) * 100
    )
  
  # Paired t-test
  tt <- t.test(paired$w1_after, paired$w1_before, paired = TRUE)
  d_values <- paired$delta
  cohens_d <- mean(d_values) / sd(d_values)
  
  cat(sprintf("  Paired t-test: t=%.3f, df=%d, p=%.6f\n",
              tt$statistic, tt$parameter, tt$p.value))
  cat(sprintf("  Mean delta: %.4f (%.1f%%)\n", mean(d_values), mean(paired$pct_change)))
  cat(sprintf("  Cohen's d: %.3f\n", cohens_d))
  cat(sprintf("  n improved: %d | n worsened: %d\n",
              sum(d_values < 0), sum(d_values > 0)))
  
  # Breakdown by country
  cat("\n  Side effects by country:\n")
  paired |>
    left_join(human |> select(persona_id, country), by = "persona_id") |>
    group_by(country) |>
    summarise(
      n = n(),
      mean_delta = mean(delta),
      mean_pct   = mean(pct_change),
      n_improved = sum(delta < 0),
      n_worsened = sum(delta > 0),
      .groups = "drop"
    ) |>
    print()
  
  cat("\n")
  
  rq3_results[[m]] <- list(
    model = m,
    paired_data = paired,
    t_stat = tt$statistic,
    df = tt$parameter,
    p_value = tt$p.value,
    mean_delta = mean(d_values),
    cohens_d = cohens_d
  )
}

# --- Bonferroni correction (4 tests: 2 models × {RQ2, RQ3}) ---
cat("  --- BONFERRONI CORRECTION (4 tests) ---\n")
all_p <- c(
  rq2_bielik = rq2_results$bielik$p_value,
  rq2_qwen   = rq2_results$qwen$p_value,
  rq3_bielik = rq3_results$bielik$p_value,
  rq3_qwen   = rq3_results$qwen$p_value
)

bonf_p <- p.adjust(all_p, method = "bonferroni")
alpha <- 0.05

bonf_table <- tibble(
  test = names(all_p),
  raw_p = all_p,
  bonferroni_p = bonf_p,
  significant = bonf_p < alpha
)

print(bonf_table)
cat("\n")

# =============================================================================
# 7. OVERLAP ANALYSIS: TOP 5 WORST BEFORE vs AFTER
# =============================================================================
cat("=" |> strrep(70), "\n")
cat("  OVERLAP ANALYSIS: WORST PERSONAS BEFORE vs AFTER\n")
cat("=" |> strrep(70), "\n\n")

for (m in c("bielik", "qwen")) {
  m_lora <- paste0(m, "_lora")
  cat(sprintf("  --- %s ---\n", toupper(m)))
  
  # Before: top 5 from baseline
  before_top5 <- top5 |>
    filter(model == m) |>
    arrange(desc(w1_norm)) |>
    pull(persona_id)
  
  # After: top 5 from posteval (ranked by W1 CI lower bound)
  after_top5 <- posteval_boot |>
    filter(model == m_lora) |>
    arrange(desc(w1_ci_lower)) |>
    head(5) |>
    pull(persona_id)
  
  overlap <- intersect(before_top5, after_top5)
  new_in_top5 <- setdiff(after_top5, before_top5)
  dropped <- setdiff(before_top5, after_top5)
  
  cat(sprintf("  Before top 5: %s\n", paste(before_top5, collapse = ", ")))
  cat(sprintf("  After top 5:  %s\n", paste(after_top5, collapse = ", ")))
  cat(sprintf("  Overlap: %d/5 (%s)\n", length(overlap),
              if (length(overlap) > 0) paste(overlap, collapse = ", ") else "none"))
  cat(sprintf("  New in top 5: %s\n",
              if (length(new_in_top5) > 0) paste(new_in_top5, collapse = ", ") else "none"))
  cat(sprintf("  Dropped out:  %s\n",
              if (length(dropped) > 0) paste(dropped, collapse = ", ") else "none"))
  
  # Show before/after for ALL original targets
  cat("\n  Before vs After for original targets:\n")
  targets_comparison <- tibble(persona_id = before_top5) |>
    left_join(
      baseline |> filter(model == m) |> select(persona_id, w1_norm) |> rename(w1_before = w1_norm),
      by = "persona_id"
    ) |>
    left_join(
      posteval_boot |> filter(model == m_lora) |> select(persona_id, w1_norm) |> rename(w1_after = w1_norm),
      by = "persona_id"
    ) |>
    mutate(
      delta = w1_after - w1_before,
      pct_change = sprintf("%+.1f%%", (delta / w1_before) * 100)
    )
  print(targets_comparison)
  
  # Show new top 5 after fine-tuning with their W1 values
  cat("\n  New top 5 after fine-tuning:\n")
  posteval_boot |>
    filter(model == m_lora) |>
    arrange(desc(w1_ci_lower)) |>
    head(5) |>
    select(persona_id, w1_norm, w1_ci_lower, w1_ci_upper) |>
    print()
  
  cat("\n")
}

# =============================================================================
# 8. SAVE ALL OUTPUTS
# =============================================================================
cat("=" |> strrep(70), "\n")
cat("  SAVING RESULTS\n")
cat("=" |> strrep(70), "\n\n")

# Posteval W1 (all 126 combinations)
out1 <- file.path(output_dir, "posteval_w1_all.csv")
posteval_boot |> write_csv(out1)
cat(sprintf("  Saved: %s\n", out1))

# Full comparison table (baseline vs posteval)
out2 <- file.path(output_dir, "baseline_vs_posteval_comparison.csv")
comparison |>
  select(persona_id, model, base_model, country, sex, age_group, education,
         base_w1, post_w1_norm, delta_w1, pct_change,
         base_ci_lower, base_ci_upper, post_w1_ci_lower, post_w1_ci_upper) |>
  write_csv(out2)
cat(sprintf("  Saved: %s\n", out2))

# RQ2 detailed results
rq2_table <- bind_rows(
  rq2_results$bielik$paired_data |> mutate(model = "bielik"),
  rq2_results$qwen$paired_data |> mutate(model = "qwen")
)
out3 <- file.path(output_dir, "rq2_worst_case_comparison.csv")
rq2_table |> write_csv(out3)
cat(sprintf("  Saved: %s\n", out3))

# RQ3 detailed results
rq3_table <- bind_rows(
  rq3_results$bielik$paired_data |> mutate(model = "bielik"),
  rq3_results$qwen$paired_data |> mutate(model = "qwen")
)
out4 <- file.path(output_dir, "rq3_side_effects_comparison.csv")
rq3_table |> write_csv(out4)
cat(sprintf("  Saved: %s\n", out4))

# Bonferroni table
out5 <- file.path(output_dir, "bonferroni_correction.csv")
bonf_table |> write_csv(out5)
cat(sprintf("  Saved: %s\n", out5))

# Summary report
out6 <- file.path(output_dir, "rq2_rq3_summary.txt")
sink(out6)
cat("=" |> strrep(70), "\n")
cat("  RQ2 + RQ3 SUMMARY REPORT\n")
cat("=" |> strrep(70), "\n\n")

cat("RQ2: Does LoRA fine-tuning reduce bias on worst-case personas?\n")
cat("=" |> strrep(50), "\n\n")
for (m in c("bielik", "qwen")) {
  r <- rq2_results[[m]]
  cat(sprintf("%s:\n", toupper(m)))
  cat(sprintf("  Targets: %s\n", paste(top5 |> filter(model == m) |> pull(persona_id), collapse = ", ")))
  cat(sprintf("  Mean W1 change: %.4f (%.1f%%)\n", r$mean_delta, r$mean_delta / mean(r$paired_data$w1_before) * 100))
  cat(sprintf("  Paired t-test: t=%.3f, df=%d, p=%.6f\n", r$t_stat, r$df, r$p_value))
  cat(sprintf("  Cohen's d: %.3f\n", r$cohens_d))
  cat(sprintf("  95%% CI: [%.4f, %.4f]\n", r$ci_lower, r$ci_upper))
  cat(sprintf("  Bonferroni p: %.6f (%s)\n\n",
              bonf_p[paste0("rq2_", m)],
              if (bonf_p[paste0("rq2_", m)] < 0.05) "SIGNIFICANT" else "not significant"))
}

cat("\nRQ3: Are there side effects on non-target personas?\n")
cat("=" |> strrep(50), "\n\n")
for (m in c("bielik", "qwen")) {
  r <- rq3_results[[m]]
  cat(sprintf("%s:\n", toupper(m)))
  cat(sprintf("  Non-target personas: %d\n", nrow(r$paired_data)))
  cat(sprintf("  Mean W1 change: %.4f (%.1f%%)\n", r$mean_delta, r$mean_delta / mean(r$paired_data$w1_before) * 100))
  cat(sprintf("  Paired t-test: t=%.3f, df=%d, p=%.6f\n", r$t_stat, r$df, r$p_value))
  cat(sprintf("  Cohen's d: %.3f\n", r$cohens_d))
  cat(sprintf("  Bonferroni p: %.6f (%s)\n\n",
              bonf_p[paste0("rq3_", m)],
              if (bonf_p[paste0("rq3_", m)] < 0.05) "SIGNIFICANT" else "not significant"))
}

cat("\nBonferroni correction table:\n")
print(bonf_table)
sink()
cat(sprintf("  Saved: %s\n", out6))

cat("\n  All outputs saved to:", output_dir, "\n")
cat("=" |> strrep(70), "\n")
cat("  ANALYSIS COMPLETE\n")
cat("=" |> strrep(70), "\n")