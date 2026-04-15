################################################################################
# Prepare LoRA Fine-Tuning Data from WVS Wave 7 Microdata
# ============================================================================
# Input:  Full WVS Wave 7 CSV (97k respondents)
# Output: Two training datasets — one for Bielik, one for Qwen
#         Aggregated mean responses per persona group per question
################################################################################

library(tidyverse)

# =============================================================================
# 0. CONFIG
# =============================================================================
WVS_PATH <- Sys.getenv("WVS_CSV_PATH", "data/raw/WVS_Cross-National_Wave_7_csv_v6_0.csv")
OUTPUT_DIR <- "./data/training"
dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)

# Countries of interest
COUNTRIES <- c("CHN", "SVK", "USA")

# 20 religious questions to EXCLUDE (prevent data leakage)
RELIGIOUS_QS <- c("Q6", "Q15", "Q64", "Q94", "Q160",
                  "Q164", "Q165", "Q166", "Q167", "Q168",
                  "Q169", "Q170", "Q171", "Q172", "Q173",
                  "Q174", "Q175", "Q239", "Q242", "Q289")

# Factual knowledge questions to EXCLUDE (not opinions)
KNOWLEDGE_QS <- c("Q91", "Q92", "Q93")

# Top 5 worst-case personas per model
BIELIK_TOP5 <- c("USA_Male_30-49_Medium", "USA_Male_18-29_Medium",
                 "USA_Male_50-64_Higher", "USA_Female_18-29_Medium",
                 "USA_Male_18-29_Higher")

QWEN_TOP5 <- c("CHN_Male_65+_Medium", "CHN_Male_65+_Lower",
               "CHN_Male_65+_Higher", "CHN_Male_50-64_Lower",
               "CHN_Male_30-49_Medium")

# =============================================================================
# 1. LOAD & FILTER WVS DATA
# =============================================================================
cat("Loading WVS Wave 7 (this may take a moment)...\n")
wvs_raw <- read_csv(WVS_PATH, show_col_types = FALSE)
cat(sprintf("  Loaded: %d rows × %d columns\n", nrow(wvs_raw), ncol(wvs_raw)))

# Filter to our 3 countries
wvs <- wvs_raw |> filter(B_COUNTRY_ALPHA %in% COUNTRIES)
cat(sprintf("  After country filter (CHN, SVK, USA): %d rows\n", nrow(wvs)))

# Check counts per country
wvs |> count(B_COUNTRY_ALPHA) |> print()

# =============================================================================
# 2. CREATE DEMOGRAPHIC GROUPS (matching human_distributions.csv)
# =============================================================================

# Sex: Q260 (1=Male, 2=Female)
wvs <- wvs |>
  mutate(
    sex = case_when(
      Q260 == 1 ~ "Male",
      Q260 == 2 ~ "Female",
      TRUE ~ NA_character_
    )
  )

# Age groups from Q262 (age in years)
wvs <- wvs |>
  mutate(
    age_group = case_when(
      Q262 >= 18 & Q262 <= 29 ~ "18-29",
      Q262 >= 30 & Q262 <= 49 ~ "30-49",
      Q262 >= 50 & Q262 <= 64 ~ "50-64",
      Q262 >= 65             ~ "65+",
      TRUE ~ NA_character_
    )
  )

# Education: Q275 uses ISCED-2011 codes (0-8)
# Lower = ISCED 0-2 (no education through lower secondary)
# Medium = ISCED 3-4 (upper secondary, post-secondary non-tertiary)
# Higher = ISCED 5-8 (tertiary and above)
wvs <- wvs |>
  mutate(
    education = case_when(
      Q275R == 1 ~ "Lower",
      Q275R == 2 ~ "Medium",
      Q275R == 3 ~ "Higher",
      TRUE ~ NA_character_
    )
  )

# Country code
wvs <- wvs |>
  mutate(country = B_COUNTRY_ALPHA)

# Build persona_id
wvs <- wvs |>
  mutate(
    persona_id = paste(country, sex, age_group, education, sep = "_")
  )

# Drop rows with missing demographics
wvs_clean <- wvs |> filter(!is.na(sex), !is.na(age_group), !is.na(education))
cat(sprintf("  After demographic filter: %d rows (dropped %d)\n",
            nrow(wvs_clean), nrow(wvs) - nrow(wvs_clean)))

# Check persona counts
cat("\n  Persona group sizes:\n")
persona_counts <- wvs_clean |>
  count(persona_id) |>
  arrange(persona_id)
cat(sprintf("  Total unique personas: %d\n", nrow(persona_counts)))
cat(sprintf("  Min respondents: %d | Max: %d | Median: %d\n",
            min(persona_counts$n), max(persona_counts$n), median(persona_counts$n)))

# =============================================================================
# 3. SELECT OPINION QUESTIONS (exclude religious, knowledge, demographics)
# =============================================================================

# All Q columns present in the data
all_q_cols <- names(wvs_clean) |> str_subset("^Q\\d+$")
cat(sprintf("\n  Total Q columns in data: %d\n", length(all_q_cols)))

# Exclude religious + knowledge
exclude_qs <- c(RELIGIOUS_QS, KNOWLEDGE_QS)
opinion_qs <- setdiff(all_q_cols, exclude_qs)

# Also exclude demographic Qs (Q260+)
demographic_qs <- opinion_qs[as.numeric(str_extract(opinion_qs, "\\d+")) >= 260]
opinion_qs <- setdiff(opinion_qs, demographic_qs)

cat(sprintf("  Religious excluded: %d\n", length(intersect(all_q_cols, RELIGIOUS_QS))))
cat(sprintf("  Knowledge excluded: %d\n", length(intersect(all_q_cols, KNOWLEDGE_QS))))
cat(sprintf("  Demographic excluded: %d\n", length(demographic_qs)))
cat(sprintf("  Opinion questions retained: %d\n", length(opinion_qs)))

# Print which religious Qs were actually found vs missing
found_religious <- intersect(all_q_cols, RELIGIOUS_QS)
missing_religious <- setdiff(RELIGIOUS_QS, all_q_cols)
if (length(missing_religious) > 0) {
  cat(sprintf("  WARNING: Religious Qs not found in data: %s\n",
              paste(missing_religious, collapse = ", ")))
}

# =============================================================================
# 4. CLEAN VALUES: Convert negative codes to NA
# =============================================================================
cat("\n  Cleaning negative codes (-1 to -5 → NA)...\n")

wvs_opinions <- wvs_clean |>
  select(persona_id, country, sex, age_group, education, all_of(opinion_qs)) |>
  mutate(across(all_of(opinion_qs), ~ ifelse(.x < 0, NA, .x)))

# Quick check: how many valid values per question
valid_pct <- wvs_opinions |>
  summarise(across(all_of(opinion_qs), ~ mean(!is.na(.x)) * 100)) |>
  pivot_longer(everything(), names_to = "question", values_to = "pct_valid") |>
  arrange(pct_valid)

cat(sprintf("  Questions with <50%% valid responses: %d\n",
            sum(valid_pct$pct_valid < 50)))
cat(sprintf("  Questions with <10%% valid responses: %d\n",
            sum(valid_pct$pct_valid < 10)))

# Remove questions with <10% valid responses (likely not asked in our countries)
drop_low_validity <- valid_pct |> filter(pct_valid < 10) |> pull(question)
if (length(drop_low_validity) > 0) {
  cat(sprintf("  Dropping %d low-validity questions: %s\n",
              length(drop_low_validity), paste(drop_low_validity, collapse = ", ")))
  opinion_qs <- setdiff(opinion_qs, drop_low_validity)
}

cat(sprintf("  Final opinion questions: %d\n", length(opinion_qs)))

# =============================================================================
# 5. AGGREGATE: Mean response per persona group per question
# =============================================================================
cat("\n  Aggregating by persona group...\n")

agg <- wvs_opinions |>
  group_by(persona_id, country, sex, age_group, education) |>
  summarise(
    n_respondents = n(),
    across(all_of(opinion_qs), ~ mean(.x, na.rm = TRUE)),
    .groups = "drop"
  )

# Replace NaN with NA (groups where ALL responses were NA for a question)
agg <- agg |> mutate(across(all_of(opinion_qs), ~ ifelse(is.nan(.x), NA, .x)))

cat(sprintf("  Aggregated to %d persona groups\n", nrow(agg)))

# =============================================================================
# 6. SPLIT INTO BIELIK AND QWEN TRAINING SETS
# =============================================================================

# Bielik: top 5 USA personas
bielik_data <- agg |> filter(persona_id %in% BIELIK_TOP5)
cat(sprintf("\n  Bielik training set: %d personas\n", nrow(bielik_data)))
if (nrow(bielik_data) != 5) {
  cat("  WARNING: Expected 5 personas for Bielik!\n")
  cat("  Missing: ", setdiff(BIELIK_TOP5, bielik_data$persona_id), "\n")
}

# Qwen: top 5 CHN personas
qwen_data <- agg |> filter(persona_id %in% QWEN_TOP5)
cat(sprintf("  Qwen training set: %d personas\n", nrow(qwen_data)))
if (nrow(qwen_data) != 5) {
  cat("  WARNING: Expected 5 personas for Qwen!\n")
  cat("  Missing: ", setdiff(QWEN_TOP5, qwen_data$persona_id), "\n")
}

# Print persona group sizes (important for reliability)
cat("\n  Bielik persona group sizes:\n")
bielik_data |> select(persona_id, n_respondents) |> print()

cat("\n  Qwen persona group sizes:\n")
qwen_data |> select(persona_id, n_respondents) |> print()

# =============================================================================
# 7. PIVOT TO LONG FORMAT (for LoRA training pair generation)
# =============================================================================

pivot_to_long <- function(df) {
  df |>
    pivot_longer(
      cols = all_of(opinion_qs),
      names_to = "question_id",
      values_to = "mean_response"
    ) |>
    filter(!is.na(mean_response)) |>
    mutate(mean_response = round(mean_response, 2)) |>
    arrange(persona_id, question_id)
}

bielik_long <- pivot_to_long(bielik_data)
qwen_long   <- pivot_to_long(qwen_data)

cat(sprintf("\n  Bielik training pairs: %d\n", nrow(bielik_long)))
cat(sprintf("  Qwen training pairs: %d\n", nrow(qwen_long)))

# =============================================================================
# 8. SAVE
# =============================================================================

# Wide format (for reference)
write_csv(bielik_data, file.path(OUTPUT_DIR, "bielik_training_wide.csv"))
write_csv(qwen_data,   file.path(OUTPUT_DIR, "qwen_training_wide.csv"))

# Long format (for LoRA script)
write_csv(bielik_long,  file.path(OUTPUT_DIR, "bielik_training_long.csv"))
write_csv(qwen_long,    file.path(OUTPUT_DIR, "qwen_training_long.csv"))

# Full aggregated data (all personas, for reference)
write_csv(agg, file.path(OUTPUT_DIR, "all_personas_aggregated.csv"))

# List of included questions
writeLines(opinion_qs, file.path(OUTPUT_DIR, "included_questions.txt"))

# List of excluded questions
writeLines(c("# Religious questions excluded (data leakage prevention):",
             RELIGIOUS_QS, "",
             "# Knowledge questions excluded (not opinions):",
             KNOWLEDGE_QS),
           file.path(OUTPUT_DIR, "excluded_questions.txt"))

cat(sprintf("\n  All files saved to: %s\n", OUTPUT_DIR))
cat("  - bielik_training_wide.csv  (5 rows × questions as columns)\n")
cat("  - bielik_training_long.csv  (persona × question pairs)\n")
cat("  - qwen_training_wide.csv    (5 rows × questions as columns)\n")
cat("  - qwen_training_long.csv    (persona × question pairs)\n")
cat("  - all_personas_aggregated.csv (all 63 personas)\n")
cat("  - included_questions.txt\n")
cat("  - excluded_questions.txt\n")

cat("\n", "=" |> strrep(70), "\n")
cat("  DATA PREPARATION COMPLETE\n")
cat("=" |> strrep(70), "\n")