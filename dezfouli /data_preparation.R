library(tidyverse)
library(magrittr)

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

df <- read_csv("raw_data.csv")

df <- df %>% 
  rename(
    id = ID,
    resp = key,
    correct = best_action
  ) %>% 
  mutate(
    id = dense_rank(id),
    resp = case_when(
      resp == "R1" ~ 1,
      resp == "R2" ~ 0
    ),
    correct_resp = case_when(
      resp == 1 & correct == TRUE ~ 1,
      resp == 1 & correct == FALSE ~ 0,
      resp == 0 & correct == TRUE ~ 0,
      resp == 0 & correct == FALSE ~ 1
    ),
    correct = as.numeric(correct)
  ) %>% 
  relocate(
    id, diag, block, correct_resp, resp, correct, reward
  )

write_csv(df, "data_prepared.csv")

sumsum <- df %>% 
  group_by(id, block) %>% 
  summarise(N = n())
