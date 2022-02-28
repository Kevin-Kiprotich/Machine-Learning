setwd('F:/BIO')
library('tidyverse')
library('GGally')
library('neuralnet')
library('dplyr')
#loading the csv
data<-read.csv('Book1.csv')

#look at the data set
ggpairs(data,title="BIOMASS DATA")

# Split into test and train sets
set.seed(13000)
data_train <- sample_frac(tbl = data, replace = FALSE, size = 0.80)
data_test <- anti_join(data, data_train)

set.seed(13000)
bio_NN1 <- neuralnet(bio2 ~ LAIL81+EVI+NDVI+DVI+SAVI, 
                       data = data,hidden =1,
                     linear.output=TRUE,algorithm='rprop+',
                     act.fct = 'logistic',err.fct = 'sse',threshold = 0.01,
                     stepmax=100000,rep=1)
plot(bio_NN1, rep = 'best')
pred<-predict(bio_NN1,data)
saveRDS(bio_NN1,file="biomodel.rds")
savedmodel<-readRDS('biomodel.rds')
pred_saved<-predict(savedmodel,data)
