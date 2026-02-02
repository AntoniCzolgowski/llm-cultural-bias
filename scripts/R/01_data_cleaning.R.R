# =============================================================================
# WVS Data Analysis - R Version
# =============================================================================

library(tidyverse)

# Load data
df <- read_csv("C:/Users/Antoni/OneDrive/Pulpit/CuBoulder/2 sem/Study/Rstudio/llm-cultural-bias/data/raw/test.csv")

# Recode variables
df <- df %>%
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
df %>% count(B_COUNTRY_ALPHA) %>% print()

# ============================================
# PART 2: Count by each demographic feature
# ============================================
cat("\n============================================================\n")
cat("BREAKDOWN BY DEMOGRAPHIC FEATURES\n")
cat("============================================================\n")

for (country in c("CHN", "SVK", "USA")) {
  cat("\n---", country, "---\n")
  country_df <- df %>% filter(B_COUNTRY_ALPHA == country)
  cat("Total N:", nrow(country_df), "\n")
  
  cat("\nSex:\n")
  country_df %>% count(SEX_LABEL) %>% print()
  
  cat("\nAge Group:\n")
  country_df %>% count(AGE_GROUP) %>% print()
  
  cat("\nEducation:\n")
  country_df %>% count(EDU_LABEL) %>% print()
}

# ============================================
# PART 3: Count all persona combinations
# ============================================
cat("\n============================================================\n")
cat("PERSONA BASKET COUNTS (Country x Sex x Age x Edu)\n")
cat("============================================================\n")

persona_counts <- df %>%
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
persona_counts %>% arrange(N) %>% head(15) %>% print()

cat("\n--- All personas sorted by N ---\n")
persona_counts %>% arrange(N) %>% print(n = Inf)

# ============================================
# PART 4: 3 Histograms (1 question x 3 countries)
# ============================================


library(tidyverse)

country_names <- c("CHN" = "China", "SVK" = "Slovakia", "USA" = "USA")

# Calculate percentages
df_plot <- df %>%
  group_by(B_COUNTRY_ALPHA) %>%
  count(Response = Q164) %>%
  mutate(
    pct = n / sum(n) * 100,
    N_total = sum(n),
    Country_Label = recode(B_COUNTRY_ALPHA, !!!country_names)
  ) %>%
  ungroup() %>%
  complete(Response = 1:10, nesting(B_COUNTRY_ALPHA, Country_Label, N_total), fill = list(n = 0, pct = 0))

# Add N to facet labels
df_plot <- df_plot %>%
  mutate(Facet_Label = paste0(Country_Label, "\n(N = ", format(N_total, big.mark = ","), ")"))

# Reorder facets
df_plot$Facet_Label <- factor(df_plot$Facet_Label, 
                              levels = c("China\n(N = 2,956)", "Slovakia\n(N = 1,065)", "USA\n(N = 2,506)"))

# Plot
p <- ggplot(df_plot, aes(x = Response, y = pct)) +
  geom_col(fill = "#4682B4", color = "white", width = 0.8) +
  geom_text(aes(label = ifelse(pct >= 3, sprintf("%.0f%%", pct), "")), 
            vjust = -0.3, size = 3, fontface = "bold") +
  facet_wrap(~ Facet_Label, nrow = 1) +
  scale_x_continuous(breaks = 1:10, expand = c(0.02, 0.02)) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.15))) +
  labs(
    x = "Response (1 = Not important, 10 = Very important)", 
    y = "Percentage (%)",
    title = "Importance of God in Life",
    subtitle = "World Values Survey Wave 7 (2017-2022)",
    caption = "Question Q164: How important is God in your life?"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold", size = 14, hjust = 0.5),
    plot.subtitle = element_text(color = "gray40", size = 10, hjust = 0.5),
    plot.caption = element_text(color = "gray50", size = 9, hjust = 0),
    strip.text = element_text(face = "bold", size = 11),
    panel.grid.major.x = element_blank(),
    panel.grid.minor = element_blank(),
    axis.title = element_text(face = "bold"),
    panel.spacing = unit(1, "lines")
  )

print(p)

ggsave("results/wvs_importance_of_god.png", p, width = 12, height = 5, dpi = 300)

# ============================================
# PART 5: Summary statistics
# ============================================
cat("\n============================================================\n")
cat("SUMMARY STATISTICS: IMPORTANCE OF GOD (Q164)\n")
cat("============================================================\n")

for (country in c("CHN", "SVK", "USA")) {
  data <- df %>% filter(B_COUNTRY_ALPHA == country) %>% pull(Q164)
  cat(sprintf("%s: Mean=%.2f, SD=%.2f, Median=%.1f, N=%d\n", 
              country, mean(data), sd(data), median(data), length(data)))
}