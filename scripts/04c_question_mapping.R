################################################################################
# Join question text mapping with LoRA training data
# ============================================================================
# Run AFTER prepare_finetuning_data.R
# Adds question_text and scale_description to long-format training files
################################################################################

library(tidyverse)

# =============================================================================
# 1. LOAD
# =============================================================================
mapping <- read_csv("data/training/wvs_question_mapping.csv", show_col_types = FALSE)
bielik  <- read_csv("data/training/bielik_training_long.csv", show_col_types = FALSE)
qwen    <- read_csv("data/training/qwen_training_long.csv", show_col_types = FALSE)

cat(sprintf("Mapping: %d questions\n", nrow(mapping)))
cat(sprintf("Bielik pairs: %d\n", nrow(bielik)))
cat(sprintf("Qwen pairs: %d\n", nrow(qwen)))

# =============================================================================
# 2. JOIN
# =============================================================================
bielik_full <- bielik |>
  left_join(mapping, by = "question_id") |>
  select(persona_id, country, sex, age_group, education, n_respondents,
         question_id, question_text, scale_description, mean_response)

qwen_full <- qwen |>
  left_join(mapping, by = "question_id") |>
  select(persona_id, country, sex, age_group, education, n_respondents,
         question_id, question_text, scale_description, mean_response)

# Check for unmatched questions
bielik_na <- sum(is.na(bielik_full$question_text))
qwen_na   <- sum(is.na(qwen_full$question_text))
cat(sprintf("\nUnmatched questions — Bielik: %d, Qwen: %d\n", bielik_na, qwen_na))

if (bielik_na > 0) {
  cat("Missing Bielik Qs: ",
      bielik_full |> filter(is.na(question_text)) |> pull(question_id) |> unique(),
      "\n")
}

# =============================================================================
# 3. PREVIEW
# =============================================================================
cat("\n=== BIELIK SAMPLE (first 5 rows) ===\n")
bielik_full |> head(5) |> print(width = 120)

cat("\n=== QWEN SAMPLE (first 5 rows) ===\n")
qwen_full |> head(5) |> print(width = 120)

# =============================================================================
# 4. SAVE
# =============================================================================
write_csv(bielik_full, "data/training/bielik_training_full.csv")
write_csv(qwen_full,   "data/training/qwen_training_full.csv")

cat(sprintf("\nSaved: bielik_training_full.csv (%d rows)\n", nrow(bielik_full)))
cat(sprintf("Saved: qwen_training_full.csv (%d rows)\n", nrow(qwen_full)))

# =============================================================================
# 5. SUMMARY STATS
# =============================================================================
cat("\n=== TRAINING DATA SUMMARY ===\n")
cat(sprintf("Bielik: %d personas × %d questions = %d training pairs\n",
            n_distinct(bielik_full$persona_id),
            n_distinct(bielik_full$question_id),
            nrow(bielik_full)))
cat(sprintf("Qwen:   %d personas × %d questions = %d training pairs\n",
            n_distinct(qwen_full$persona_id),
            n_distinct(qwen_full$question_id),
            nrow(qwen_full)))






















#Filter, need more than 5 responses in an aggregated persona profile


# ============================================================
# Filter: minimum 5 valid responses per question per persona
# ============================================================

# We need to go back to individual-level data to count valid responses per question per persona
# Reload the cleaned WVS data and count non-NA per persona × question

wvs_opinions <- read_csv("data/training/all_personas_aggregated.csv", show_col_types = FALSE)

# The aggregated file has means but not per-question N counts
# We need to recompute from raw data

wvs_raw <- read_csv(Sys.getenv("WVS_CSV_PATH", "data/raw/WVS_Cross-National_Wave_7_csv_v6_0.csv"),
                    show_col_types = FALSE)

COUNTRIES <- c("CHN", "SVK", "USA")
opinion_qs <- read_lines("data/training/included_questions.txt")

wvs <- wvs_raw |>
  filter(B_COUNTRY_ALPHA %in% COUNTRIES) |>
  mutate(
    sex = case_when(Q260 == 1 ~ "Male", Q260 == 2 ~ "Female"),
    age_group = case_when(
      Q262 >= 18 & Q262 <= 29 ~ "18-29",
      Q262 >= 30 & Q262 <= 49 ~ "30-49",
      Q262 >= 50 & Q262 <= 64 ~ "50-64",
      Q262 >= 65 ~ "65+"
    ),
    education = case_when(Q275R == 1 ~ "Lower", Q275R == 2 ~ "Medium", Q275R == 3 ~ "Higher"),
    persona_id = paste(B_COUNTRY_ALPHA, sex, age_group, education, sep = "_")
  ) |>
  filter(!is.na(sex), !is.na(age_group), !is.na(education)) |>
  select(persona_id, all_of(opinion_qs)) |>
  mutate(across(all_of(opinion_qs), ~ ifelse(.x < 0, NA, .x)))

# Count valid (non-NA) responses per persona × question
valid_counts <- wvs |>
  group_by(persona_id) |>
  summarise(across(all_of(opinion_qs), ~ sum(!is.na(.x))), .groups = "drop") |>
  pivot_longer(-persona_id, names_to = "question_id", values_to = "n_valid")

cat(sprintf("Total persona × question combos: %d\n", nrow(valid_counts)))
cat(sprintf("Combos with n_valid < 5: %d\n", sum(valid_counts$n_valid < 5)))
cat(sprintf("Combos with n_valid == 0: %d\n", sum(valid_counts$n_valid == 0)))

# Join with training data and filter
MIN_VALID <- 5

bielik_full <- read_csv("data/training/bielik_training_full.csv", show_col_types = FALSE)
qwen_full   <- read_csv("data/training/qwen_training_full.csv", show_col_types = FALSE)

cat(sprintf("\nBEFORE filter — Bielik: %d, Qwen: %d\n", nrow(bielik_full), nrow(qwen_full)))

bielik_filtered <- bielik_full |>
  left_join(valid_counts, by = c("persona_id", "question_id")) |>
  filter(n_valid >= MIN_VALID) |>
  select(-n_valid)

qwen_filtered <- qwen_full |>
  left_join(valid_counts, by = c("persona_id", "question_id")) |>
  filter(n_valid >= MIN_VALID) |>
  select(-n_valid)

cat(sprintf("AFTER filter (n_valid >= %d) — Bielik: %d, Qwen: %d\n",
            MIN_VALID, nrow(bielik_filtered), nrow(qwen_filtered)))
cat(sprintf("Dropped — Bielik: %d, Qwen: %d\n",
            nrow(bielik_full) - nrow(bielik_filtered),
            nrow(qwen_full) - nrow(qwen_filtered)))

# Per-persona breakdown
cat("\n=== Per-persona question counts ===\n")
cat("BIELIK:\n")
bielik_filtered |> count(persona_id) |> print()
cat("\nQWEN:\n")
qwen_filtered |> count(persona_id) |> print()

# Save filtered versions
write_csv(bielik_filtered, "data/training/bielik_training_full.csv")
write_csv(qwen_filtered,   "data/training/qwen_training_full.csv")
cat("\nOverwritten bielik_training_full.csv and qwen_training_full.csv\n")