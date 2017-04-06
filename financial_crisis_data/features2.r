library(tidyverse)
library(readxl)
library(countrycode)
library(TTR)
library(zoo)
library(xts)

# read and process data
jst <- read_excel("JSTdatasetR1.xlsx",sheet = "Data") %>% 
  select(-country)%>% rename(Year = year)
crises <- read_excel("AE_Crisesdates2.xlsx", sheet = "CrisesDatesCombined") %>%
  mutate(iso = countrycode(.$Country, "country.name", "iso3c") ) %>%
  filter(iso %in% jst$iso) %>%
  select(iso,everything())

# Join data
#crises_jst1 <- crises %>% full_join(jst, by = c("Year","iso"))
crises_jst2 <- crises %>% right_join(jst, by = c("Year","iso"))

# define compute streak
computeStreak <- function(priceSeries) {
  signs <- sign(diff(priceSeries))
  posDiffs <- negDiffs <- rep(0,length(signs))
  posDiffs[signs == 1] <- 1
  negDiffs[signs == -1] <- -1
  
  # Create vector of cumulative sums and cumulative
  # sums not incremented during streaks.
  # Zero out any leading NAs after na.locf
  posCum <- cumsum(posDiffs)
  posNAcum <- posCum
  posNAcum[posDiffs == 1] <- NA
  posNAcum <- na.locf(posNAcum, na.rm = FALSE)
  posNAcum[is.na(posNAcum)] <- 0
  posStreak <- posCum - posNAcum
  
  # Repeat for negative cumulative sums
  negCum <- cumsum(negDiffs)
  negNAcum <- negCum
  negNAcum[negDiffs == -1] <- NA
  negNAcum <- na.locf(negNAcum, na.rm = FALSE)
  negNAcum[is.na(negNAcum)] <- 0
  negStreak <- negCum - negNAcum
  
  streak <- posStreak + negStreak
  streak
  #streak <- xts(streak, order.by = index(priceSeries))
  return (streak)
}

# define connorsRSI
connorsRSI <- function(price, nRSI = 3, nStreak = 2,nPercentLookBack = 10) {
  priceRSI <- RSI(price, nRSI)
  #print(length(priceRSI))
  streakRSI <- RSI(c(NA,computeStreak(price)), nStreak)
  #print(length(streakRSI))
  percents <- c(NA,round(runPercentRank(x = diff(log(price)),
                                        n = nPercentLookBack, cumulative = FALSE,
                                        exact.multiplier = 1) * 100))
  #print(length(percents))
  ret <- (priceRSI + streakRSI + percents) / 3
  #colnames(ret) <- "connorsRSI"
  return(ret)
}

# Feature_6
crises_jst2 <- crises_jst2 %>%group_by(iso)%>%
  mutate(no.crises.p.3yrs = crisis.tally == 0 & lag(crisis.tally == 0) & lag(crisis.tally == 0,n = 2)& lag(crisis.tally == 0,n = 3)) %>%
  mutate(no.crises.p.3yrs = ifelse(no.crises.p.3yrs == T,1,0) )

# Feature_7
crises_jst2 <- crises_jst2 %>% group_by(iso) %>% 
  mutate(ca.deficit.neg3yrs = ifelse(ca < 0 & lead(ca) < 0 & lead(ca,2) < 0,1,0))

# Feature_8
crises_jst2 <- crises_jst2 %>% group_by(iso) %>% 
  mutate(yield.conversion = ifelse(stir > ltrate,1,0))

# Feautre_9
crises_jst2 <- crises_jst2 %>% group_by(iso) %>% 
  mutate(stir.up = ifelse(stir > lag(stir),1,0))

# Feature_10
crises_jst2 <- crises_jst2 %>% group_by(iso) %>% 
  mutate(impexp.ratio = lag(imports/exports))
