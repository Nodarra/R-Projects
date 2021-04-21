###Classification of heart diseased patients###

###Libraries###
library(class)
library(C50)
library(rpart)
library(rpart.plot)
library(gmodels)
library(ggplot2)
library(reshape2)
library(ISLR)
library(caret)
library(pROC)
library(ROCR)
library(neuralnet)
library(tidyverse)
library(plotrix)
library(randomForest)
library(kernlab)
library(naivebayes)

###Importing data set###

heart <- read.csv("C:/Users/sisic/OneDrive/Desktop/Dataset/heart.csv", stringsAsFactors = FALSE)

###Exploring data set###

str(heart)

###Changing variable name###

names(heart)[1] <- "age"

###Checking for missing and zero values and omitting them###

heart <- na.omit(heart)

heart <- heart[heart$thal !=0, ]

heart <- heart[heart$ca !=4, ]

###Factoring data###

heart2 <- heart

heart$sex <- factor(heart$sex, levels = c("1", "0"), labels = c("Male", "Female"))

heart$target <- factor(heart$target, levels = c("1", "0"), labels = c("Healthy", "Diseased"))

heart$slope <- factor(heart$slope, levels =c("2", "1", "0"), labels = c("up", "flat", "down")) ###SLOPE OF PEAK 

heart$fbs <- factor(heart$fbs, levels = c("0", "1"), labels = c("<=120", ">120")) ###FASTING BLOOD SUGAR###

heart$thal <- factor(heart$thal, levels = c("3", "2", "1"), labels = c("fixed defect", "normal", "reversable defect"))

heart$restecg <- factor(heart$restecg, levels = c("2", "1", "0"), labels = c("hypertrophy", "normal", "wave abnormality")) ###RESTING ELCTROCARDIOGRAHIC RESULTS###

heart$cp <- factor(heart$cp, levels = c("3", "2", "1", "0"), labels = c("typical angina", "non-anginal", "atypical angina", "asymp angina")) ###CHEST PAIN###

heart$exang <- factor(heart$exang, levels = c("1", "0"), labels = c("Yes", "No")) 

summary(heart)

###Visualizing data###

###Healthy v Diseased retio###
ggplot(heart, aes(target, fill=target)) + geom_bar() + labs(x="Disease", y="Number of patients") + guides(fill=FALSE)

###Disease occurance in sex groups###
ggplot(heart, aes(sex, fill=target)) + geom_bar() + labs(fill="Diagnosis", x="sex", y="Number of patients")

###Chest pain distribution###
ggplot(heart, aes(cp, fill=target)) + geom_bar() + labs(fill="Diagnosis", x="Pain in chest", y="Number of patients")

###Blood pressure###

ggplot(heart, aes(trestbps, fill=target)) + geom_histogram(binwidth=5) + labs(fill="Diagnosis", x="Blood pressure", y="Number of patients")


###Age###

ggplot(heart, aes(age, fill=target)) +  geom_histogram(binwidth=1) + labs(fill="Diagnosis", x="Age", y="Number of patients")

###Sugar in blood levels###

ggplot(heart, aes(fbs, fill=target)) + geom_bar() + labs(fill="Diagnosis", x="Sugar level", y="Number of patients")

###Presence of pain during exercise###

ggplot(heart, aes(exang, fill=target)) + geom_bar() + labs(fill="Diagnosis", x="Pain during exercise", y="Number of patients")


heart[heart$thalach < 100, c("age", "thalach", "target")]

heart[heart$thalach > 180, c("age", "thalach", "target")]

###Splitting data into training and test set###

set.seed(1234)

rows <- sample(nrow(heart))

heart <- heart[rows, ]

data_split <- createDataPartition(heart$target, p = 0.7, list = FALSE)

train_set <- heart[data_split, ]

test_set <- heart[-data_split, ]

###10 fold cross validation###

fitControl <- trainControl(method="cv", number=10)

###Logistic regression###

LR_model <- train(target ~ ., data = train_set, method="glm", family=binomial())

prediction_LR <- predict(LR_model, test_set)

confusionMatrix(prediction_LR, test_set$target)

###Decision tree###

DT_model <- train(target ~., data = train_set, method="rpart")

prediction_DT <- predict(DT_model, test_set)

confusionMatrix(prediction_DT, test_set$target)

rpart.plot(DT_model$finalModel)

###Naive Bayes###

NB_model <- train(target ~., data = train_set, method="naive_bayes")

prediction_NB <- predict(NB_model, test_set)

confusionMatrix(prediction_NB, test_set$target)

###Random forest### 

RF_model <- train(target ~., data = train_set, method="rf")

prediction_RF <- predict(RF_model, test_set)

confusionMatrix(prediction_RF, test_set$target)


