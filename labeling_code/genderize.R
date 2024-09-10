library(predictrace)
library(dplyr)
setwd("/home/nisse/Documents/vakken_l/Thesis/nih/originals/july/data/final_files")

df <- read.csv("processed_combined_aligned_name_split.csv")

df$predicted_gender <- NA
result <- predict_gender(df$PI_FIRST_NAME, probability = FALSE)

# Extracting the likely_race column
likely_gender_column <- result$likely_gender
# Adding the new column to the dataframe
df$predicted_gender <- likely_gender_column

# Correcting the typo in the column name
df$predicted_gender[is.na(df$predicted_gender)] <- "unknown"
df <- mutate(df, predicted_gender = case_when(
  predicted_gender == "male" ~ "m",
  predicted_gender == "female" ~ "f",
  TRUE ~ predicted_gender  # Keep other values unchanged
))
df <- df %>%
  rename("PI_GENDER" = colnames(.)[ncol(.)])
head(df)
write.csv(df, file = "genderized1.csv", row.names = FALSE)