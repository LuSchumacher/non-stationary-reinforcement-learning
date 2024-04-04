library(tidyverse)
library(magrittr)

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

df <- read_csv("Fontanesi2019.csv")

df$pair_type <- NA
df$pair_type[df$cor_option == 2] <- 1
df$pair_type[df$cor_option == 3] <- 2
df$pair_type[df$cor_option == 4 & df$inc_option == 2] <- 3
df$pair_type[df$cor_option == 4 & df$inc_option == 3] <- 4

df %<>% 
  mutate(resp = ifelse(accuracy == 1, cor_option, inc_option)) %>% 
  select(participant, trial_block, block_label,
         rt, resp, accuracy,
         f_cor:inc_option, pair_type) %>% 
  rename(id = participant,
         correct = accuracy,
         trial = trial_block,
         block = block_label) %>% 
  mutate(resp = resp - 1,
         cor_option = cor_option - 1,
         inc_option = inc_option - 1, 
         pair_type = pair_type - 1)

# write_csv(df, "data_fontanesi_prep.csv")

sumsum <- df %>% 
  group_by(id) %>% 
  summarise(n = length(rt))
