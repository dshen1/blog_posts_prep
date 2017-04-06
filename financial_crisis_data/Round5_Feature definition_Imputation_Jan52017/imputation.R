library(mice)
library(Amelia)
library(tidyverse)
library(forecast)
library(readxl)
library(countrycode)
library(imputeTS)

# read and process data
jst <- read_excel("JSTdatasetR1.xlsx",sheet = "Data") %>% 
  select(-country)%>% rename(Year = year)
crises <- read_excel("AE_Crisesdates2.xlsx", sheet = "CrisesDatesCombined") %>%
  mutate(iso = countrycode(.$Country, "country.name", "iso3c") ) %>%
  filter(iso %in% jst$iso) %>%
  select(iso,everything())

# Join data
crises_jst2 <- crises %>% right_join(jst, by = c("Year","iso"))

#mice
crises_jst3 <- crises_jst2 %>%
  select(iso,Year,hpnom,xrusd,stocks,cpi,rgdpmad)

imputed_Data <- crises_jst3 %>% group_by(iso) %>%
  mice(., m=5, maxit = 50, method = 'pmm', seed = 500)
completedata.mice <- mice::complete(imputed_Data,5)

completedata.mice$hpnom1 = crises_jst2$hpnom
completedata.mice$xrusd1 = crises_jst2$xrusd
completedata.mice$stocks1 = crises_jst2$stocks

ggplot(data = completedata.mice)+geom_line(aes(x = Year,y = hpnom))+
  geom_line(aes(x = Year,y = hpnom1),colour = "red")+
  facet_wrap( ~ iso, ncol=3)+
  ggtitle(expression(atop("House Prices", atop(italic("imputed data in black (using MICE)"), ""))))
  

ggplot(data = completedata.mice)+geom_line(aes(x = Year,y = xrusd))+
  geom_line(aes(x = Year,y = xrusd1),colour = "red")+
  facet_wrap( ~ iso, ncol=3)+
  ggtitle(expression(atop("Exchange Rate", atop(italic("imputed data in black (using MICE)"), ""))))

ggplot(data = completedata.mice)+geom_line(aes(x = Year,y = stocks))+
  geom_line(aes(x = Year,y = stocks1),colour = "red")+
  facet_wrap( ~ iso, ncol=3)+
  ggtitle(expression(atop("Stocks", atop(italic("imputed data in black (using MICE)"), ""))))

# Amelia
crises_jst4 <- crises_jst2 %>%
  select(iso,Year,hpnom,stocks,cpi,rgdpmad,iy,debtgdp,ltrate)

completedata.amelia<-data.frame()
for(i in levels(factor(crises_jst4$iso))){
  temp <- amelia(subset(crises_jst4,iso == i)[,-1],m=5, parallel = "multicore")
  completedata.amelia<- rbind(completedata.amelia,temp$imputations$imp5)
}


completedata.amelia$hpnom1 = crises_jst2$hpnom
completedata.amelia$stocks1 = crises_jst2$stocks
completedata.amelia$iso = crises_jst2$iso

ggplot(data = completedata.amelia)+geom_line(aes(x = Year,y = hpnom))+
  geom_line(aes(x = Year,y = hpnom1),colour = "red")+
  facet_wrap( ~ iso, ncol=3)+
  ggtitle(expression(atop("House Prices", atop(italic("imputed data in black (using Amelia)"), ""))))


# ggplot(data = imputed_data_amelia)+geom_line(aes(x = Year,y = xrusd))+
#   geom_line(aes(x = Year,y = xrusd1),colour = "red")+
#   facet_wrap( ~ iso, ncol=3)+
#   ggtitle(expression(atop("Exchange Rate", atop(italic("imputed data in black (using Amelia)"), ""))))

ggplot(data = completedata.amelia)+geom_line(aes(x = Year,y = stocks))+
  geom_line(aes(x = Year,y = stocks1),colour = "red")+
  facet_wrap( ~ iso, ncol=3)+
  ggtitle(expression(atop("Stocks", atop(italic("imputed data in black (using Amelia)"), ""))))


#interpolation
interp <- function(x){
  x[which(!is.na(x))[1]:length(x)] <- na.interp(x[which(!is.na(x))[1]:length(x)])
  return(x)
  }

imputation<-function(x,y){ #x is incomplete data for a country, y is the code for the country
  x1<-x[,-c(1:2)] # remove first 2 columns
  imp.x1<- apply(x1,2, interp) #impute data for each column
  imp.x1<-as.data.frame(imp.x1) 
  imp.x1<-data.frame(x[,1:2],imp.x1) # add removed columns back
  return(imp.x1) # return imputed data for the country
}


crises_jst4 <- crises_jst2%>%
  select(iso,Year,hpnom,xrusd,stocks)

completedata.interp<-data.frame()
for(i in levels(factor(crises_jst4$iso))){
  completedata.interp<- rbind(completedata.interp,imputation(filter(crises_jst4, iso == i),i))
}

completedata.interp$hpnom1 = crises_jst2$hpnom
completedata.interp$xrusd1 = crises_jst2$xrusd
completedata.interp$stocks1 = crises_jst2$stocks

ggplot(data = completedata.interp)+geom_line(aes(x = Year,y = hpnom))+
  geom_line(aes(x = Year,y = hpnom1),colour = "red")+
  facet_wrap( ~ iso, ncol=3)+
  ggtitle(expression(atop("House Prices", atop(italic("imputed data in black (using interpolation)"), ""))))

ggplot(data = completedata.interp)+geom_line(aes(x = Year,y = xrusd))+
  geom_line(aes(x = Year,y = xrusd1),colour = "red")+
  facet_wrap( ~ iso, ncol=3)+
  ggtitle(expression(atop("Exchange Rate", atop(italic("imputed data in black (using interpolation)"), ""))))

ggplot(data = completedata.interp)+geom_line(aes(x = Year,y = stocks))+
  geom_line(aes(x = Year,y = stocks1),colour = "red")+
  facet_wrap( ~ iso, ncol=3)+
  ggtitle(expression(atop("Stocks", atop(italic("imputed data in black (using interpolation)"), ""))))



#splines interpolation
imputation.splines<-function(x,y){ #x is incomplete data for a country, y is the code for the country
  x1<-x[,-c(1:2)] # remove first 2 columns
  imp.x1<- apply(x1,2, function(a) na.interpolation(a, option = "spline")) #impute data for each column
  imp.x1<-as.data.frame(imp.x1) 
  imp.x1<-data.frame(x[,1:2],imp.x1) # add removed columns back
  return(imp.x1) # return imputed data for the country
}


crises_jst5 <- crises_jst2%>%
  select(iso,Year,hpnom,xrusd,stocks)

completedata.splines<-data.frame()
for(i in levels(factor(crises_jst5$iso))){
  completedata.splines<- rbind(completedata.splines,imputation.splines(filter(crises_jst5, iso == i),i))
}

completedata.splines$hpnom1 = crises_jst2$hpnom
completedata.splines$xrusd1 = crises_jst2$xrusd
completedata.splines$stocks1 = crises_jst2$stocks

ggplot(data = completedata.splines)+geom_line(aes(x = Year,y = hpnom))+
  geom_line(aes(x = Year,y = hpnom1),colour = "red")+
  facet_wrap( ~ iso, ncol=3)+
  ggtitle(expression(atop("House Prices", atop(italic("imputed data in black (using splines interpolation)"), ""))))

ggplot(data = completedata.splines)+geom_line(aes(x = Year,y = xrusd))+
  geom_line(aes(x = Year,y = xrusd1),colour = "red")+
  facet_wrap( ~ iso, ncol=3)+
  ggtitle(expression(atop("Exchange Rate", atop(italic("imputed data in black (using splines interpolation)"), ""))))

ggplot(data = completedata.splines)+geom_line(aes(x = Year,y = stocks))+
  geom_line(aes(x = Year,y = stocks1),colour = "red")+
  facet_wrap( ~ iso, ncol=3)+
  ggtitle(expression(atop("Stocks", atop(italic("imputed data in black (using splines interpolation)"), ""))))


# Plots 
stocks.mice <- completedata.mice %>%
  select(stocks,iso,Year)%>%
  mutate(method = vector(mode = "character",length = nrow(.)))%>%
  mutate(method = "mice")

stocks.amelia <- completedata.amelia %>%
  select(stocks,iso,Year)%>%
  mutate(method = vector(mode = "character",length = nrow(.)))%>%
  mutate(method = "amelia")

stocks.interp <- completedata.interp %>%
  select(stocks,iso,Year)%>%
  mutate(method = vector(mode = "character",length = nrow(.)))%>%
  mutate(method = "Interpolation")

stocks.splines <- completedata.splines %>%
  select(stocks,iso,Year)%>%
  mutate(method = vector(mode = "character",length = nrow(.)))%>%
  mutate(method = "Splines")

stocks <- bind_rows(stocks.mice,stocks.amelia,stocks.interp,stocks.splines)

ggplot(data = stocks)+
  geom_line(aes(x = Year,y = stocks, colour = method))+
  facet_wrap( ~ iso, ncol=3, scales = "free_y")+ggtitle("Stocks")

# hpnorm

hpnom.mice <- completedata.mice %>%
  select(hpnom,iso,Year)%>%
  mutate(method = vector(mode = "character",length = nrow(.)))%>%
  mutate(method = "mice")

hpnom.amelia <- completedata.amelia %>%
  select(hpnom,iso,Year)%>%
  mutate(method = vector(mode = "character",length = nrow(.)))%>%
  mutate(method = "amelia")

hpnom.interp <- completedata.interp %>%
  select(hpnom,iso,Year)%>%
  mutate(method = vector(mode = "character",length = nrow(.)))%>%
  mutate(method = "Interpolation")

hpnom.splines <- completedata.splines %>%
  select(hpnom,iso,Year)%>%
  mutate(method = vector(mode = "character",length = nrow(.)))%>%
  mutate(method = "Splines")

dat.hpnom <- bind_rows(hpnom.mice,hpnom.amelia,hpnom.interp,hpnom.splines)

ggplot(data = dat.hpnom)+
  geom_line(aes(x = Year,y = hpnom, colour = method))+
  facet_wrap( ~ iso, ncol=3, scales = "free_y")+ggtitle("Housing Prices")

# xrusd
xrusd.mice <- completedata.mice %>%
  select(xrusd,iso,Year)%>%
  mutate(method = vector(mode = "character",length = nrow(.)))%>%
  mutate(method = "mice")

# xrusd.amelia <- completedata.amelia %>%
#   select(xrusd,iso,Year)%>%
#   mutate(method = vector(mode = "character",length = nrow(.)))%>%
#   mutate(method = "amelia")

xrusd.interp <- completedata.interp %>%
  select(xrusd,iso,Year)%>%
  mutate(method = vector(mode = "character",length = nrow(.)))%>%
  mutate(method = "Interpolation")

xrusd.splines <- completedata.splines %>%
  select(xrusd,iso,Year)%>%
  mutate(method = vector(mode = "character",length = nrow(.)))%>%
  mutate(method = "Splines")

dat.xrusd <- bind_rows(xrusd.mice,xrusd.interp,xrusd.splines)

ggplot(data = dat.xrusd)+
  geom_line(aes(x = Year,y = xrusd, colour = method))+
  facet_wrap( ~ iso, ncol=3,scales = "free_y")+ggtitle("Exhange Rate")
