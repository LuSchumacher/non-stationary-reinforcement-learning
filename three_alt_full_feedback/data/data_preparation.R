library(tidyverse)
library(magrittr)

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

df <- read_csv("empiric_data.csv")

summary <- df %>% 
  group_by(condition) %>% 
  summarise(N = n())


summary <- df %>% 
  group_by(condition) %>% 
  summarise(
    mean_low = mean(low_opt_feed),
    sd_low = sd(low_opt_feed),
    min_low = min(low_opt_feed),
    max_low = max(low_opt_feed),
    mean_mid = mean(mid_opt_feed),
    sd_mid = sd(mid_opt_feed),
    mean_high = mean(high_opt_feed),
    sd_high = sd(high_opt_feed),
    N = n()
  )


x = round(rnorm(2093, 42, 5))
sd(x)
