library(tidyverse)
library(magrittr)
library(rstan)
library(bayesplot)
library(cmdstanr)

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

df <- read_csv("../data/TortoiseAndHareData.csv") %>% 
  filter(
    phase == 0
  ) %>% 
  rename(
    id = ID,
    block = learningblock,
    set_size = ns,
    correct_resp = corchoice,
    resp = choice,
    correct = cor,
    iteration = iter
  ) %>% 
  select(-pcor, -delay, -phase) %>% 
  mutate(
    id = dense_rank(id),
    block = block,
    stim = stim -1,
    resp = resp - 1,
    correct_resp = correct_resp -1,
    reward = ifelse(correct == 1, 1, 0)
  )

summary <- df %>% 
  group_by(id) %>% 
  summarise(
    n = length(rt),
    n_blocks = max(block)
  )

write_csv(df, "data_prepared.csv")

person_data <- df %>% 
  filter(id == 1)



