setwd("C:/Users/brend/Dropbox/Macro/master_strategy/features/Round5_Feature definition_Imputation_Jan52017")

library(tidyverse)
library(readxl)
library(countrycode)
library(TTR)
library(zoo)
library(xts)
library(mice)
library(forecast)

#read and process data 
jst <- read_excel("JSTdatasetR1.xlsx", sheet = "Data") %>% 
  select(-country) %>% rename(Year = year)
View(jst) 
crises <- read_excel("AE_Crisesdates2.xlsx", sheet = "CrisesDatesCombined") %>% 
  mutate(iso = countrycode(.$Country, "country.name", "iso3c") ) %>% 
  filter(iso %in% jst$iso) %>% 
    select(iso, everything()) 

#join data 
crisis_jst2 <- crises %>% right_join(jst, by = c("Year", "iso"))

#define compute streak, function used in connors RSI 
computeStreak <- function(priceSeries) {
  signs <- sign(diff(priceSeries))
  posDiffs <- negDiffs <- rep(0, length(signs))
  posDiffs[signs == 1] <- 1
  neegDiffs[signs == 1] <- 1
}

Questions
#1. Line 16 - Refresh my memory, I know you mentioned some manual work here, but 
#how were the crises dates combined in the sheet? 

Next Steps:
#1. Event studys
#2. Time Series graphs 
#3. Probabilities 

