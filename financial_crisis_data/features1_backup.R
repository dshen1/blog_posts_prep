#data import and features 
#loaad data and packages

#setwd("C:/Users/Home/Dropbox/Macro")
setwd("C:/Users/nonyabusiness/Dropbox/Macro")

library(psych)
library(ggplot2)
library(plotly)
library(dplyr)
library(tidyr)
library(XLConnect)
library(quantstrat)
library(stringi)

#Plotly - Someday - create an .Rprofile document 
Sys.setenv("plotly_username"="historysquared")
Sys.setenv("plotly_api_key"="e92vklsaij")

#read and format data 
#st.panel <- tbl_df(read.csv("C:/Users/Home/Dropbox/Macro/schularick_taylor_RRcrises_panel.csv"))
st.panel <- tbl_df(read.csv("C:/Users/nonyabusiness/Dropbox/Macro/schularick_taylor_RRcrises_panel.csv", strip.white = TRUE))
st.panel$year <- paste(st.panel$year, "-12-31")
st.panel$year <- stri_replace_all_fixed(st.panel$year, " ", "")
st.panel$year <- as.POSIXct(strptime(st.panel$year, format = "%Y-%m-%d", tz = "GMT"))
class(st.panel$year)
#create percent function 
percent <- function(x, digits = 2, format = "f", ...) {
  paste0(formatC(100 * x, format = format, digits = digits, ...), "%")
}

##FEATURE DEFINITIONS 
#interpolate data with linear  
#http://stackoverflow.com/questions/27920690/linear-interpolation-using-dplyr
#maxgap allows up to four consecutive NAs
#rule option allows extrapololation into the flanking time points
st.panel = st.panel %>% 
  group_by(iso) %>% 
  arrange(iso, year) %>% 
  mutate(time = seq(1, n())) %>% 
  mutate(loans1.linear = approx(time, loans1, time)$y) %>%
  mutate(loans1.spline = spline(time, loans1, n=n())$y) %>% 
  select(-time) %>% 
  ungroup()
#One and Five year ROC for loans1 and stocks 
st.panel = st.panel %>%
  group_by(iso) %>% 
  arrange(iso, year) %>%   
  mutate(loans1.ROC1 = ROC(loans1, n = 1)) %>% 
  mutate(loans1.ROC5 = ROC(loans1, n = 5)) %>%
  mutate(stocks.ROC1 = ROC(stocks, n = 1)) %>% 
  mutate(stocks.ROC5 = ROC(stocks, n = 5)) %>% 
  ungroup()

#Lags 
st.panel = st.panel %>%
  group_by(iso) %>% 
  arrange(iso, year) %>%   
  mutate(loans1.ROC1.lag1 = lag(loans1.ROC1, n = 1)) %>% 
  mutate(loans1.ROC5.lag1 = lag(loans1.ROC5, n = 1)) %>%
  mutate(stocks.ROC1.lag1 = lag(stocks.ROC1, n = 1)) %>% 
  mutate(stocks.ROC5.lag1 = lag(stocks.ROC5, n = 1)) %>% 
  ungroup()
View(st.panel)

#Deciles
st.panel  <- st.panel %>%
  group_by(iso) %>%
  arrange(iso, year) %>%
  mutate(loans1.ROC1.lag1.decile = ntile(loans1.ROC1.lag1, 10)) %>% 
  mutate(loans1.ROC5.lag1.decile = ntile(loans1.ROC5.lag1, 10))  %>%
  mutate(stocks.ROC1.lag1.decile = ntile(stocks.ROC1.lag1, 10)) %>% 
  mutate(stocks.ROC5.lag1.decile = ntile(stocks.ROC5.lag1, 10)) %>% 
  ungroup()
describe(st.panel)
#**check to see 

st.panel.deciles <- st.panel %>%
  select(iso, year, loans1, loans1.ROC1.lag1, loans1.ROC5.lag1, 
         loans1.ROC1.lag1.decile, loans1.ROC5.lag1.decile, stocks.ROC1.lag1,
         stocks.ROC5.lag1, stocks.ROC1.lag1.decile, stocks.ROC5.lag1.decile, currency,
         inflation, stock_market, domestic_debt, external_debt,
         banking_crisis, crisis_tally)  
st.panel.deciles

#gather data
st.panel.deciles <- st.panel %>%
  select(iso, year, loans1, loans1.ROC1.lag1, loans1.ROC5.lag1, 
         loans1.ROC1.lag1.decile, loans1.ROC5.lag1.decile, currency,
         inflation, stock_market, domestic_debt, external_debt,
         banking_crisis, crisis_tally) %>% 
  gather(crisis_type, yes_no, currency:banking_crisis)
names(st.panel.deciles)

#**Next steps: ImputeTS package -- http://rpackages.ianhowson.com/cran/imputeTS/
#**Add back in NAs for stock market crashes 
       