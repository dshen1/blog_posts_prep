setwd("C:/Users/brend/Dropbox/Macro/master_strategy/data")

library(tidyverse)
library(readxl)
library(countrycode)
library(TTR)
library(zoo)
library(xts)
library(mice)
library(forecast)


# read and process data
jst <- read_excel("JSTdatasetR1.xlsx",sheet = "Data") %>% 
  select(-country)%>% rename(Year = year)
crises <- read_excel("AE_Crisesdates2.xlsx", sheet = "CrisesDatesCombined") %>%
  mutate(iso = countrycode(.$Country, "country.name", "iso3c") ) %>%
  filter(iso %in% jst$iso) %>%
  select(iso,everything())

# Join data
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

# Feature_2
crises_jst2 <- crises_jst2 %>% group_by(iso) %>% 
  mutate(tloanspch4 = ifelse((((tloans-lag(tloans,4))/lag(tloans,4))*100)>40,1,0))

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

# Feature_9
crises_jst2 <- crises_jst2 %>% group_by(iso) %>% 
  mutate(stir.up = ifelse(stir > lag(stir),1,0))


# Feature_10
crises_jst2 <- crises_jst2 %>% group_by(iso) %>% 
  mutate(impexp.ratio = lag(imports/exports))


###########################################################################
###########################################################################
# IMPUTATION FOR THE VARIABLES HPNOM, XRUSD, STOCKS using interpolation
# for (Features 4, 5 and 3), flanks omitted

#interpolation

interp.no.flanks <- function(x){
  x[which(!is.na(x))[1]:length(x)] <- na.interp(x[which(!is.na(x))[1]:length(x)])
  return(x)
}

imputation<-function(x,y){ #x is incomplete data for a country, y is the code for the country
  x1<-x[,-c(1:2)] # remove first 2 columns
  imp.x1<- apply(x1,2,interp.no.flanks) #impute data for each column
  imp.x1<-as.data.frame(imp.x1) 
  imp.x1<-data.frame(x[,1:2],imp.x1) # add removed columns back
  return(imp.x1) # return imputed data for the country
}


completeData2 <- crises_jst2%>%
  select(iso,Year,hpnom,xrusd,stocks,tloans,gdp)

imputed_data<-data.frame()
for(i in levels(factor(completeData2$iso))){
  imputed_data<- rbind(imputed_data,imputation(filter(completeData2, iso == i),i))
}

imputed_data <- imputed_data%>%
  mutate(hpnom1 = crises_jst2$'hpnom',xrusd1 = crises_jst2$'xrusd',stocks1 = crises_jst2$'stocks',tloans1 = crises_jst2$'tloans',gdp1 = crises_jst2$'gdp')


ggplot(data = imputed_data)+geom_line(aes(x = Year,y = hpnom))+
  geom_line(aes(x = Year,y = hpnom1),colour = "red")+
  facet_wrap( ~ iso, ncol=3)+
  ggtitle(expression(atop("House Prices", atop(italic("imputed data in black (using interpolation)"), ""))))

ggplot(data = imputed_data)+geom_line(aes(x = Year,y = xrusd))+
  geom_line(aes(x = Year,y = xrusd1),colour = "red")+
  facet_wrap( ~ iso, ncol=3)+
  ggtitle(expression(atop("Exchange Rate", atop(italic("imputed data in black (using interpolation)"), ""))))

ggplot(data = imputed_data)+geom_line(aes(x = Year,y = stocks))+
  geom_line(aes(x = Year,y = stocks1),colour = "red")+
  facet_wrap( ~ iso, ncol=3)+
  ggtitle(expression(atop("Stocks", atop(italic("imputed data in black (using interpolation)"), ""))))


ggplot(data = imputed_data)+geom_line(aes(x = Year,y = tloans))+
  geom_line(aes(x = Year,y = tloans1),colour = "red")+
  facet_wrap( ~ iso, ncol=3)+
  ggtitle(expression(atop("Tloans", atop(italic("imputed data in black (using interpolation)"), ""))))

ggplot(data = imputed_data)+geom_line(aes(x = Year,y = gdp))+
  geom_line(aes(x = Year,y = gdp1),colour = "red")+
  facet_wrap( ~ iso, ncol=3)+
  ggtitle(expression(atop("gdp", atop(italic("imputed data in black (using interpolation)"), ""))))
######################################################################
################################################################3#####

## Continuing with imputed data from interpoplation
crises_jst2 <- ungroup(crises_jst2)%>%
  mutate(hpnom = imputed_data$'hpnom', stocks = imputed_data$'stocks', xrusd = imputed_data$'xrusd')%>%
  mutate(gdp = imputed_data$'gdp', tloans = imputed_data$'tloans')


# Feature 3
# crises_jst2 <- crises_jst2 %>% group_by(iso)%>%       #dplyr doesn't work
#   mutate(stocks.crsi = as.numeric(connorsRSI(.$stocks)))

stocks.crsi = c()
for(i in levels(factor(crises_jst2$iso))){
  stocks.crsi <- c(stocks.crsi,connorsRSI(crises_jst2$stocks[crises_jst2$iso==i]))
}

crises_jst2$stocks.crsi <- stocks.crsi


# Feature 4
# crises_jst2 <- crises_jst2 %>% 
#   mutate(hpnom.crsi = connorsRSI(.$hpnom)) #dplyr doesn't work


hpnom.crsi = c()
for(i in levels(factor(crises_jst2$iso))){
  hpnom.crsi <- c(hpnom.crsi,connorsRSI(crises_jst2$hpnom[crises_jst2$iso==i]))
}
crises_jst2$hpnom.crsi <- hpnom.crsi

# Feature 5
# crises_jst2 <- crises_jst2 %>%
#   mutate(xrusd.crsi = connorsRSI(.$xrusd)) #dplyr doesn't work

xrusd.crsi = c()
for(i in levels(factor(crises_jst2$iso))){
  xrusd.crsi <- c(xrusd.crsi,connorsRSI(crises_jst2$xrusd[crises_jst2$iso==i]))
}
crises_jst2$xrusd.crsi <- xrusd.crsi

## Feature 1

# ma.credit.gdp.crsi <- c()
# for(i in levels(factor(crises_jst2$iso))){
#   credit.gdp <- ma(crises_jst2$tloans[crises_jst2$iso==i]/crises_jst2$gdp[crises_jst2$iso==i],10)
#   ma.credit.gdp.crsi <- c(ma.credit.gdp.crsi,connorsRSI(credit.gdp))
# }

# crises_jst2$xrusd.crsi <- xrusd.crsi
# crises_jst2 <- crises_jst2 %>% mutate(credit.gdp = tloans/gdp) %>%
#   group_by(iso) %>%
#   mutate(credit.gdp = ma(.$credit.gdp, order=10))


credit.gdp.crsi <- c()
 for(i in levels(factor(crises_jst2$iso))){
   credit.gdp <- crises_jst2$tloans[crises_jst2$iso==i]/crises_jst2$gdp[crises_jst2$iso==i]
   credit.gdp.crsi <- c(credit.gdp.crsi,connorsRSI(credit.gdp))
}

crises_jst2 <- crises_jst2 %>% mutate(credit.gdp.crsi = credit.gdp.crsi)

##
ggplot(data = crises_jst2)+geom_line(aes(x = Year,y = credit.gdp.crsi))+
#  geom_line(aes(x = Year,y = gdp1),colour = "red")+
  facet_wrap( ~ iso, ncol=3)+ggtitle("Credit/GDP CRSI")
#  ggtitle(expression(atop("gdp", atop(italic("imputed data in black (using interpolation)"), ""))))


