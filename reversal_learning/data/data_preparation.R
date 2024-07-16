library(tidyverse)
library(magrittr)

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

load('data_exp2.RData')

df <- dat %>%
  tibble() %>% 
  rename(
    id = pp,
    trial = TrialNumber,
    stim_set = stimulus_set,
    p_b = p_win_correct,
    p_a = p_win_incorrect,
    choice = choiceIsHighP,
  ) %>% 
  select(
    id, block, trial, reversal, stim_set,
    p_a, p_b, choice, reward
  )

df$stim_set[df$block == 2] <- df$stim_set[df$block == 2] - 2
df$stim_set[df$block == 3] <- df$stim_set[df$block == 3] - 4
df$stim_set[df$block == 4] <- df$stim_set[df$block == 4] - 6

df$p_a_tmp <- df$p_a
df$p_b_tmp <- df$p_b
df$p_a[df$reversal == 0] <- df$p_a_tmp[df$reversal == 0]
df$p_a[df$reversal == 1] <- df$p_b_tmp[df$reversal == 1]
df$p_b[df$reversal == 0] <- df$p_b_tmp[df$reversal == 0]
df$p_b[df$reversal == 1] <- df$p_a_tmp[df$reversal == 1]

df <- df %>% 
  select(-p_a_tmp, -p_b_tmp)

write_csv(df, 'data_prepared.csv')