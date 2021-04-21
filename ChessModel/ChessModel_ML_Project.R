###CHESS MODEL###
###Predicting the outcome of a chess game###

###LIBRARIES###

library(ggplot2)
library(dplyr)
library(rpart)
library(rpart.plot)
library(class)
library(C50)
library(gmodels)
library(reshape2)
library(ISLR)
library(caret)
library(pROC)
library(ROCR)
library(tidyverse)
library(plotrix)
library(randomForest)
library(kernlab)
library(naivebayes)

###Importing data set###

games <- read.csv("C:/Users/sisic/OneDrive/Desktop/ML_Project/games.csv", stringsAsFactors = FALSE)

###Exploring data set### 

str(games)

head(games$winner)

tail(games)

###Checking for zero and NA values and omitting them###

lapply(games,function(x) { length(which(is.na(x)))})

lapply(games,function(x) { length(which(is.null(x)))})

###Data preparation###

#Removing games where players run out of time#

games <- games[games$victory_status != "outoftime", ]

#Removing games which ended in a draw#

games <- games[games$winner != "draw", ]

#Filtering out games that are not rated and where the rating difference is less than 1000#

games <- filter(games, (abs(white_rating - black_rating) < 1000) | (abs(white_rating - black_rating) > 1000 & rated == TRUE))

#Filtering games where rating is above 1000#

games <- filter(games, white_rating > 1000 & black_rating > 1000)

#Proportion of target attribute#

table(games$winner)

###Data visualization###

mean(games$white_rating)

mean(games$black_rating)

ggplot(data = games) + geom_histogram(aes(x=white_rating), fill = "white") + geom_histogram(aes(x=black_rating), fill = "black", alpha = 0.2)

ggplot(games, aes(winner, fill=winner)) + geom_bar() + labs(x="Winner", y="Number of games") + guides(fill=FALSE) +  scale_fill_manual(values = c("white" = "white",
                                                                                                                                                  "black" = "black"))
#Separating individual moves#

games$moves <- strsplit(as.character(games$moves), " ")

#Getting rid of data set inconsistency#

rows <- nrow(games)

for(i in 1:rows){
  if(games$rated[i] == "False" | games$rated[i] == "FALSE"){
    games$rated[i] <- FALSE
  }
  else{
    games$rated[i] <- TRUE
  }
}


#Separating white and black moves#


for(i in 1:rows){
  moves <- unlist(games$moves[i])
  
  games$w_moves[i]<- paste(moves[c(TRUE, FALSE)], collapse = " ")
  
  games$b_moves[i]<- paste(moves[c(FALSE, TRUE)], collapse = " ")
}

games$w_moves <- strsplit((games$w_moves), " ")

games$b_moves <- strsplit((games$b_moves), " ")

#Defining new variables that indicate if the player castled or not#

games$w_castled <- FALSE 

games$b_castled <- FALSE

#Scanning black and white moves for castling#

for(i in 1:rows){
  if("O-O" %in% unlist(games$w_moves[i]) | "O-O-O" %in% unlist(games$w_moves[i])){
    games$w_castle[i]<- TRUE
  }
  if("O-O" %in% unlist(games$b_moves[i]) | "O-O-O" %in% unlist(games$b_moves[i])){
    games$b_castle[i]<- TRUE
  }
}

#Determining if only one or both players have castled#

for(i in 1:rows){
  if (games$w_castle[i] == TRUE & games$b_castle[i]== FALSE){
    games$castle[i] <- "white"
  }
  else if (games$w_castle[i] == FALSE & games$b_castle[i]== TRUE){
    games$castle[i] <- "black"
  }
  else if (games$w_castle[i] == TRUE & games$b_castle[i]== TRUE){
    games$castle[i] <- "both"
  }
  else if (games$w_castle[i] == FALSE & games$b_castle[i]== FALSE){
    games$castle[i] <- "none"
  }
}

#Determining who is controlling more important squares#

#White#
for(i in 1:rows){
  
  score <- 0
  
  for(n in 1:length(unlist(games$w_moves[i]))){
    
    moves <- unlist(games$w_moves[i])[n]

      if (grepl("d4", moves, fixed = TRUE) == TRUE){
        score <- score + 1
      }
      else if (grepl("e4", moves, fixed = TRUE) == TRUE){
        score <- score + 1
      }
      else if (grepl("d5", moves, fixed = TRUE) == TRUE){
        score <- score + 1
      }
      else if (grepl("e5", moves, fixed = TRUE) == TRUE){
        score <- score + 1
      }
  }
  games$w_control[i] <- score
}
  
#Black#
for(i in 1:rows){
  
  score <- 0
  
  for(n in 1:length(unlist(games$b_moves[i]))){
    
    moves <- unlist(games$b_moves[i])[n]

      if (grepl("d4", moves, fixed = TRUE) == TRUE){
        score <- score + 1
      }
      else if (grepl("e4", moves, fixed = TRUE) == TRUE){
        score <- score + 1
      }
      else if (grepl("d5", moves, fixed = TRUE) == TRUE){
        score <- score + 1
      }
      else if (grepl("e5", moves, fixed = TRUE) == TRUE){
        score <- score + 1
      }
  }
  games$b_control[i] <- score
}


#Determining opening in the game#

for(i in 1:nrow(games)){
  
  opening <- as.character(games$opening_eco[i])
  
  if(grepl("A",opening,fixed=TRUE) == TRUE){
    games$open[i] <- "A"
  }
  else if (grepl("B",opening,fixed=TRUE) == TRUE){
    games$open[i] <- "B"
  }
  else if (grepl("C",opening,fixed=TRUE) == TRUE){
    games$open[i] <- "C"
  }
  else if (grepl("D",opening,fixed=TRUE) == TRUE){
    games$open[i] <- "D"
  }
  else{
    games$open[i] <- "E"
  }
}

###Training attributes###

games$control <- games$w_control - games$b_control 

games$rating <- games$white_rating - games$black_rating

###Separating data into training set and test set###

set.seed(1234)

rows <- sample(nrow(games))

games <- games[rows, ]

split <- createDataPartition(games$winner, p = 0.8, list = FALSE)

train_set <- games[split, ] 

test_set <- games[-split, ]

###Defining validation###

fitControl <- trainControl(method = "cv", number = 10, classProbs = TRUE, summaryFunction = twoClassSummary, savePredictions = T)

###Defining accuracy function###

accuracy <- function(ground_truth, predictions) {
  mean(ground_truth == predictions)
}

###########################################################

###Creating Machine learning models###

###Decision tree###

DT_model <- train(winner ~ rating+castle+control+open, data = train_set, method = "rpart", metric = "ROC", trControl = fitControl)

#Plotting the tree#

rpart.plot(DT_model$finalModel)

#Evaluating model performance#

DT_predictions <- predict(DT_model, test_set)

confusionMatrix(as.factor(DT_predictions), as.factor(test_set$winner))

#Plotting ROC Curve#

DT_predictions_prob <- predict(DT_model, test_set, type = "prob")

DT_pred <- prediction(as.numeric(as.factor(DT_predictions_prob[,2])), as.numeric(as.factor(test_set$winner)))

DT_performance <- performance(DT_pred, "tpr","fpr")

plot(DT_performance,main = "ROC Curve for decision tree",col = 2,lwd = 2) + abline(a = 0,b = 1,lwd = 2,lty = 3,col = "black")

#Calculating Area Under the Curve#

DT_AUC <- performance(DT_pred, measure = "auc")

DT_AUC@y.values[[1]]

###########################################################

###Naive Bayes###

NB_model <- train(winner ~ rating+castle+control+open, data = train_set, method = "naive_bayes", trControl = fitControl)

#Evaluating model performance#

NB_predictions <- predict(NB_model, test_set)

confusionMatrix(NB_predictions, as.factor(test_set$winner))

#Plotting ROC curve#

NB_predictions_prob <- predict(NB_model, test_set, type = "prob")

NB_pred <- prediction(as.numeric(NB_predictions_prob[,2]), as.numeric(as.factor(test_set$winner)))

NB_performance <- performance(NB_pred, "tpr","fpr")

plot(NB_performance,main = "ROC Curve for Naive Bayes",col = 2,lwd = 2) + abline(a = 0,b = 1,lwd = 2,lty = 3,col = "black")

#Calculating Area Under the Curve#

NB_AUC <- performance(NB_pred, measure = "auc")

NB_AUC@y.values[[1]]

###########################################################

#KNN#

KNN_model <- train(winner ~ rating+castle+control+open, method = "knn", trControl = fitControl, data = train_set)

#Evaluating model performance#

KNN_predictions <- predict(KNN_model, test_set)

confusionMatrix(KNN_predictions, as.factor(test_set$winner))

#Plotting ROC curve#

KNN_predictions_prob <- predict(KNN_model, test_set, type = "prob")

KNN_pred <- prediction(as.numeric(KNN_predictions_prob[,2]), as.numeric(as.factor(test_set$winner)))

KNN_performance <- performance(KNN_pred, "tpr","fpr")

plot(KNN_performance,main = "ROC Curve for KNN",col = 2,lwd = 2) + abline(a = 0,b = 1,lwd = 2,lty = 3,col = "black")

#Calculating Area Under the Curve#

KNN_AUC <- performance(KNN_pred, measure = "auc")

KNN_AUC@y.values[[1]]

##########################################################

#Logistic Regression#

LR_model <-  train(winner ~ rating+castle+control+open, data = train_set, method = "glm", trControl = fitControl)

#Evaluating model performance#

LR_predictions <- predict(LR_model, test_set)

confusionMatrix(LR_predictions, as.factor(test_set$winner))

#Plotting ROC curve#

LR_predictions_prob <- predict(LR_model, test_set, type = "prob")

LR_pred <- prediction(as.numeric(LR_predictions_prob[,2]), as.numeric(as.factor(test_set$winner)))

LR_performance <- performance(LR_pred, "tpr","fpr")

plot(LR_performance,main = "ROC Curve for Logistic Regression",col = 2,lwd = 2) + abline(a = 0,b = 1,lwd = 2,lty = 3,col = "black")

#Calculating Area Under the Curve#

LR_AUC <- performance(LR_pred, measure = "auc")

LR_AUC@y.values[[1]]

#########################################################






