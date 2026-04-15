# =============================================================================
# WVS Data Validation & Persona Counts
# =============================================================================
# Loads the WVS Wave 7 subset (CHN, SVK, USA) and validates demographic
# distributions for the 63-persona design (Country x Sex x Age x Education).
# =============================================================================

library(tidyverse)

# Load data
df <- read_csv("data/raw/wvs_wave7_subset.csv", show_col_types = FALSE)

# Recode variables
df <- df |>
  mutate(
    AGE_GROUP = case_when(
      Q262 >= 18 & Q262 <= 29 ~ "18-29",
      Q262 >= 30 & Q262 <= 49 ~ "30-49",
      Q262 >= 50 & Q262 <= 64 ~ "50-64",
      Q262 >= 65 ~ "65+"
    ),
    AGE_GROUP = factor(AGE_GROUP, levels = c("18-29", "30-49", "50-64", "65+")),
    SEX_LABEL = factor(Q260, levels = c(1, 2), labels = c("Male", "Female")),
    EDU_LABEL = factor(Q275R, levels = c(1, 2, 3), labels = c("Lower", "Middle", "Higher"))
  )

# ============================================
# PART 1: Count observations per country
# ============================================
cat("============================================================\n")
cat("TOTAL OBSERVATIONS PER COUNTRY\n")
cat("============================================================\n")
df |> count(B_COUNTRY_ALPHA) |> print()

# ============================================
# PART 2: Count by each demographic feature
# ============================================
cat("\n============================================================\n")
cat("BREAKDOWN BY DEMOGRAPHIC FEATURES\n")
cat("============================================================\n")

for (country in c("CHN", "SVK", "USA")) {
  cat("\n---", country, "---\n")
  country_df <- df |> filter(B_COUNTRY_ALPHA == country)
  cat("Total N:", nrow(country_df), "\n")

  cat("\nSex:\n")
  country_df |> count(SEX_LABEL) |> print()

  cat("\nAge Group:\n")
  country_df |> count(AGE_GROUP) |> print()

  cat("\nEducation:\n")
  country_df |> count(EDU_LABEL) |> print()
}

# ============================================
# PART 3: Count all persona combinations
# ============================================
cat("\n============================================================\n")
cat("PERSONA BASKET COUNTS (Country x Sex x Age x Edu)\n")
cat("============================================================\n")

persona_counts <- df |>
  count(B_COUNTRY_ALPHA, SEX_LABEL, AGE_GROUP, EDU_LABEL, name = "N")

cat("\nTotal unique personas:", nrow(persona_counts), "\n")
cat("Expected max: 3 x 2 x 4 x 3 =", 3*2*4*3, "\n")
cat("\nPersonas with N < 10:", sum(persona_counts$N < 10), "\n")
cat("Personas with N < 30:", sum(persona_counts$N < 30), "\n")
cat("Personas with N < 50:", sum(persona_counts$N < 50), "\n")
cat("Personas with N >= 50:", sum(persona_counts$N >= 50), "\n")

cat("\nMin N:", min(persona_counts$N), "\n")
cat("Max N:", max(persona_counts$N), "\n")
cat("Mean N:", round(mean(persona_counts$N), 1), "\n")
cat("Median N:", median(persona_counts$N), "\n")

cat("\n--- Smallest persona baskets ---\n")
persona_counts |> arrange(N) |> head(15) |> print()

cat("\n--- All personas sorted by N ---\n")
persona_counts |> arrange(N) |> print(n = Inf)

# ============================================
# PART 4: Summary statistics
# ============================================
cat("\n============================================================\n")
cat("SUMMARY STATISTICS: IMPORTANCE OF GOD (Q164)\n")
cat("============================================================\n")

for (country in c("CHN", "SVK", "USA")) {
  data <- df |> filter(B_COUNTRY_ALPHA == country) |> pull(Q164)
  cat(sprintf("%s: Mean=%.2f, SD=%.2f, Median=%.1f, N=%d\n",
              country, mean(data), sd(data), median(data), length(data)))
}
